#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, argparse
from typing import List, Tuple, Optional
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from tqdm.auto import tqdm

# ===================== Utils =====================
def set_seed(seed:int, strict_det:bool=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = strict_det
    torch.backends.cudnn.benchmark = not strict_det

def split_indices(n:int, tr:float, vr:float, seed:int)->Tuple[List[int],List[int],List[int]]:
    idx=list(range(n)); random.Random(seed).shuffle(idx)
    ntr=int(n*tr); nvr=int(n*vr)
    return idx[:ntr], idx[ntr:ntr+nvr], idx[ntr+nvr:]

# ============ Persistent H5 helpers ============
def _attach_h5_recursive(ds):
    """Attach a persistent h5 file handle to P2CDataset inside (possibly nested) Subset."""
    while isinstance(ds, Subset):
        ds = ds.dataset
    if isinstance(ds, P2CDataset) and getattr(ds, "_h5", None) is None:
        ds._h5 = h5py.File(ds.h5, "r", swmr=True, libver="latest")

def _worker_init_fn(_worker_id):
    info = torch.utils.data.get_worker_info()
    _attach_h5_recursive(info.dataset)

# ===================== Dataset =====================
class P2CDataset(Dataset):
    """
    Probes (time window) -> Centerline band at given z (single time t).

    Input:
      probes_seq: (T_window, 9, 3) -> flatten -> (T_window*9, 3)
    Target:
      centerline: (Ny, Nx_target, C_out)  # C_out = len(target_comps)

    归一化：
     - /meta/section_velocity_mean (3,), /meta/section_velocity_std (3,)
     - probes 始终 3 通道；target 仅所选通道
    """
    def __init__(
        self,
        h5:str,
        time_window:int=21,
        z_start_1based:int=21,
        nx_target:int=21,
        x_start:Optional[int]=None,
        augment:bool=False,
        time_stride:int=1,
        normalize:bool=True,
        eps:float=1e-6,
        target_comp_idx: Optional[List[int]] = None,
    ):
        assert time_window % 2 == 1, "time_window must be odd"
        assert time_stride >= 1 and nx_target >= 1
        self.h5, self.tw, self.half = h5, time_window, time_window//2
        self.z0, self.augment, self.tstride = max(0, z_start_1based-1), augment, time_stride
        self.nx_target, self.x_start = nx_target, x_start
        self.items=[]; self._h5=None

        # 目标通道：默认 xyz
        t_idx = [0,1,2] if not target_comp_idx else sorted(set(int(i) for i in target_comp_idx))
        if any(i not in (0,1,2) for i in t_idx): raise ValueError("target_comp_idx 仅允许 0/1/2")
        self.target_comp_idx = t_idx

        # 读取 meta 统计
        self.normalize = normalize
        self.eps = float(eps)
        self.norm_mean_probes = np.zeros(3, dtype=np.float32)
        self.norm_std_probes  = np.ones(3, dtype=np.float32)
        self.norm_mean_target = np.zeros(len(self.target_comp_idx), dtype=np.float32)
        self.norm_std_target  = np.ones (len(self.target_comp_idx), dtype=np.float32)
        try:
            with h5py.File(h5, "r") as f:
                if normalize and "meta" in f and \
                   "section_velocity_mean" in f["meta"] and "section_velocity_std" in f["meta"]:
                    m = f["meta"]["section_velocity_mean"][...].astype(np.float32).reshape(3)
                    s = np.maximum(f["meta"]["section_velocity_std"][...].astype(np.float32).reshape(3), self.eps)
                    self.norm_mean_probes[:] = m; self.norm_std_probes[:] = s
                    self.norm_mean_target[:] = m[self.target_comp_idx]; self.norm_std_target[:] = s[self.target_comp_idx]
                else:
                    if normalize: print("[Normalize] meta/{section_velocity_mean,std} 未找到，禁用归一化。")
                    self.normalize = False
        except Exception as e:
            if normalize: print(f"[Normalize] 读取 meta 失败（{e}），禁用归一化。")
            self.normalize = False

        # 构建样本索引
        with h5py.File(h5,"r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 missing /cases")
            for cname,g in f["cases"].items():
                if not isinstance(g,h5py.Group) or ("section" not in g or "probes" not in g): continue
                Ts,Zs,Ny,Xs,Cs = g["section"].shape
                Tp,Zp,P9,Cp = g["probes"].shape
                if not (Ts==Tp and Zs==Zp and Cp==3 and Cs==3 and P9==9):
                    raise RuntimeError(f"{cname}: shape mismatch - section {g['section'].shape}, probes {g['probes'].shape}")

                xs = (Xs - self.nx_target)//2 if self.x_start is None else int(self.x_start)
                xe = xs + self.nx_target
                if xs<0 or xe>Xs: raise RuntimeError(f"{cname}: Invalid X slice [{xs}:{xe}) for Xs={Xs}")
                self._xs, self._xe, self._ny = xs, xe, Ny

                for z in range(self.z0, Zs):
                    t1, t2 = self.half, Ts-1-self.half
                    if t1>t2: break
                    for t in range(t1, t2+1, self.tstride):
                        self.items.append((cname,z,t))
        if not self.items:
            raise RuntimeError("No valid (case,z,t); check time_window/z_start/time_stride.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname,z,t=self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5,"r")
        g=f["cases"][cname]

        p = g["probes"][t-self.half:t+self.half+1, z, :, :].astype(np.float32)               # (T,9,3)
        s = g["section"][t, z, :, self._xs:self._xe, :].astype(np.float32)[..., self.target_comp_idx]  # (Ny,nx,C)

        if self.augment:
            if random.random()<0.5: s = s[:, ::-1, :]
            if random.random()<0.5: s = s[::-1, :, :]; p = p[:, ::-1, :]

        if self.normalize:
            p = (p - self.norm_mean_probes) / self.norm_std_probes
            s = (s - self.norm_mean_target) / self.norm_std_target

        p = torch.from_numpy(np.ascontiguousarray(p.reshape(-1, 3)))  # (T*9,3)
        s = torch.from_numpy(np.ascontiguousarray(s))                  # (Ny,nx,C)

        if self._h5 is None: f.close()
        return {"probes_seq":p, "target":s, "meta":{"case":cname,"z":z,"t":t}}

def collate(batch):
    x=torch.stack([b["probes_seq"] for b in batch],0)
    y=torch.stack([b["target"] for b in batch],0)
    return {"probes_seq":x, "target":y, "meta":[b["meta"] for b in batch]}

# ===================== Model =====================
class PosEnc(nn.Module):
    def __init__(self,d_model:int,max_len:int=20000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div[:-1] if d_model%2==1 else pos*div)
        self.register_buffer("pe",pe.unsqueeze(1))
    def forward(self,x):  # x: (L,N,D)
        return x + self.pe[:x.size(0)]

class P2CTransformer(nn.Module):
    def __init__(self,time_window:int, ny:int=49, nx_target:int=21,
                 d_model:int=160, nhead:int=5, enc_layers:int=4, dec_layers:int=2,
                 ffn:int=640, drop:float=0.1, act:str="gelu", out_c:int=3):
        super().__init__()
        self.seq_len=time_window*9
        self.out_y,self.out_x,self.out_c=ny,nx_target,out_c
        nq=ny*nx_target

        self.in_proj=nn.Linear(3,d_model)
        self.pos=PosEnc(d_model, max_len=self.seq_len+8)
        enc = nn.TransformerEncoderLayer(d_model,nhead,ffn,drop,activation=act,batch_first=False,norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model,nhead,ffn,drop,activation=act,batch_first=False,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,enc_layers)
        self.decoder=nn.TransformerDecoder(dec,dec_layers)
        self.q = nn.Parameter(torch.randn(nq,d_model))
        self.qpos = nn.Parameter(torch.zeros(nq,d_model)); nn.init.normal_(self.qpos, std=0.02)
        self.head=nn.Linear(d_model,out_c)

    def forward(self,x):  # x: (B,L,3)
        B,L,C=x.shape; assert L==self.seq_len and C==3
        h=self.in_proj(x).transpose(0,1)     # (L,B,D)
        h=self.encoder(self.pos(h))          # (L,B,D)
        q=(self.q+self.qpos).unsqueeze(1).expand(-1,B,-1)   # (nq,B,D)
        y=self.decoder(q,h).transpose(0,1)   # (B,nq,D)
        return self.head(y).view(B,self.out_y,self.out_x,self.out_c)

# ================= Train / Eval =====================
def enable_flash_attention():
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception:
        pass

def train_one_epoch(model, loader, optim, device, epoch:int, epochs:int, scaler, amp_dtype, amp_on:bool, wb):
    model.train(); mse=nn.MSELoss(); running=0.0; n=0
    pbar=tqdm(loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for batch in pbar:
        x=batch["probes_seq"].to(device, non_blocking=True)
        y=batch["target"].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else torch.cpu.amp.autocast(enabled=False)
        with ctx:
            loss=mse(model(x),y)
        if amp_on:
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        else:
            loss.backward(); optim.step()
        b=x.size(0); running+=loss.item()*b; n+=b
        avg = running/max(1,n)
        pbar.set_postfix(loss=f"{avg:.4e}")
        if wb is not None: wb.log({"train/loss": loss.item(), "train/avg_loss": avg, "epoch": epoch}, commit=True)
    return running/max(1,n)

@torch.no_grad()
def evaluate(model, loader, device, tag="Val", epoch:int=None, epochs:int=None,
             amp_dtype=None, amp_on:bool=False, wb=None, ny:int=49, nx:int=21, nc:int=3):
    model.eval(); smse=smae=0.0; n=0
    desc = f"{tag}" if epoch is None else f"{tag} {epoch+1}/{epochs}"
    pbar = tqdm(loader, desc=desc, leave=False)
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else torch.cpu.amp.autocast(enabled=False)
    for batch in pbar:
        x = batch["probes_seq"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with ctx:
            yhat = model(x)
        smse += F.mse_loss(yhat, y, reduction="sum").item()
        smae += F.l1_loss (yhat, y, reduction="sum").item()
        n += x.size(0)
        denom = max(1, n * ny * nx * nc)
        pbar.set_postfix(MSE=f"{(smse/denom):.4e}", MAE=f"{(smae/denom):.4e}")
    denom = max(1, n * ny * nx * nc)
    m, a = smse/denom, smae/denom
    if wb is not None and epoch is not None: wb.log({f"{tag.lower()}/MSE_norm": m, f"{tag.lower()}/MAE_norm": a, "epoch": epoch}, commit=True)
    return m, a

def save_ckpt(path, model, optim, epoch, args, best=None, last_only=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":optim.state_dict(),"epoch":epoch,"args":vars(args),"best_val":best}, path)
    if not last_only:
        torch.save(model.state_dict(), os.path.join(os.path.dirname(path),"model_only.pt"))

def load_ckpt(path, model, optim=None, map_location="cpu"):
    ckpt=torch.load(path,map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optim is not None and "optimizer" in ckpt: optim.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch",0)+1, ckpt.get("best_val",None)

# ================== wandb utils ==================
def init_wandb(args, model):
    if not args.wandb: return None
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] import 失败: {e}\n使用 --no-wandb 关闭或 pip install wandb"); return None
    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or None,
                     mode=args.wandb_mode, config=vars(args), resume="allow", save_code=True)
    try: wandb.watch(model, log="gradients", log_freq=args.wandb_watch_freq, log_graph=False)
    except Exception: pass
    return wandb

# ===================== Main =====================
def _parse_target_comps(s: str)->List[int]:
    m={'x':0,'y':1,'z':2}
    s=(s or "xyz").lower().strip()
    idx=sorted(set(m[c] for c in s if c in m))
    if not idx: raise ValueError(f"--target-comps '{s}' 无有效通道，示例：x / yz / xyz")
    return idx

def parse_args():
    p=argparse.ArgumentParser("Transformer: probes(time-window) -> centerline(t)")
    p.add_argument("--h5", required=True)
    p.add_argument("--save-dir", default="./runs/probes2centerline")
    p.add_argument("--resume", default="")
    p.add_argument("--eval-only", action="store_true")

    # dataset / sampling
    p.add_argument("--time-window", type=int, default=21)
    p.add_argument("--z-start-1based", type=int, default=21)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--nx-target", type=int, default=21)
    p.add_argument("--x-start", type=int, default=None)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--samples-per-epoch", type=int, default=0)
    p.add_argument("--val-samples", type=int, default=0)
    p.add_argument("--test-samples", type=int, default=0)

    # normalization
    p.add_argument("--no-normalize", action="store_true")

    # 选择回归分量
    p.add_argument("--target-comps", type=str, default="xyz",
                   help="回归目标分量：x / y / z / xy / yz / xz / xyz（默认 xyz）")

    # run
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)

    # optim
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    # model
    p.add_argument("--d-model", type=int, default=160)
    p.add_argument("--nhead", type=int, default=5)
    p.add_argument("--num-enc-layers", type=int, default=4)
    p.add_argument("--num-dec-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=640)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    # perf toggles
    p.add_argument("--amp", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true")

    # device & GPU selection
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu", type=int, default=-1)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="probes2centerline")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online","offline","disabled"])
    p.add_argument("--wandb-watch-freq", type=int, default=200)
    return p.parse_args()

def maybe_subsample_subset(subset_obj, k:int, seed:int):
    if k<=0 or k>=len(subset_obj): return subset_obj
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(subset_obj), size=k, replace=False).tolist())
    return Subset(subset_obj.dataset, [subset_obj.indices[i] for i in pick])

def resolve_device(args):
    if args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available():
        n = torch.cuda.device_count()
        dev = f"cuda:{args.gpu if args.gpu < n else 0}"
    else:
        dev = args.device
        if dev.startswith("cuda") and not torch.cuda.is_available(): dev = "cpu"
    print(f"[Device] Using {dev}")
    return dev

def _clone_split_with_stats(base: P2CDataset, idx: List[int], normalize_flag: bool):
    """用训练集统计量克隆一个数据集并按 idx 子集化。"""
    ds = P2CDataset(
        base.h5, base.tw, base.z0+1, nx_target=base.nx_target, x_start=base.x_start,
        augment=False, time_stride=base.tstride, normalize=normalize_flag,
        target_comp_idx=base.target_comp_idx
    )
    # 强制与训练一致的 mean/std
    ds.norm_mean_probes[:] = base.norm_mean_probes
    ds.norm_std_probes[:]  = np.maximum(base.norm_std_probes, 1e-6)
    ds.norm_mean_target[:] = base.norm_mean_target
    ds.norm_std_target[:]  = np.maximum(base.norm_std_target, 1e-6)
    return Subset(ds, idx)

def main():
    args=parse_args()
    target_idx = _parse_target_comps(args.target_comps)

    set_seed(args.seed, strict_det=not args.cudnn_benchmark)
    torch.set_float32_matmul_precision("high")

    # device
    args.device = resolve_device(args)
    if args.device.startswith("cuda"): enable_flash_attention()

    # datasets（训练）
    d_train_full=P2CDataset(
        args.h5, args.time_window, args.z_start_1based,
        nx_target=args.nx_target, x_start=args.x_start,
        augment=True, time_stride=args.time_stride,
        normalize=(not args.no_normalize),
        target_comp_idx=target_idx,
    )
    ny, nx, nc = d_train_full._ny, d_train_full.nx_target, len(target_idx)
    N=len(d_train_full); tr_idx, va_idx, te_idx = split_indices(N, args.train_ratio, args.val_ratio, args.seed)
    d_train=Subset(d_train_full, tr_idx)

    # datasets（验证/测试），重用训练统计
    d_val  = _clone_split_with_stats(d_train_full, va_idx, d_train_full.normalize)
    d_test = _clone_split_with_stats(d_train_full, te_idx, d_train_full.normalize)
    d_val  = maybe_subsample_subset(d_val,  args.val_samples,  args.seed+1)
    d_test = maybe_subsample_subset(d_test, args.test_samples, args.seed+2)

    # loaders
    common = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
                  persistent_workers=(args.num_workers>0), prefetch_factor=args.prefetch_factor,
                  worker_init_fn=_worker_init_fn)
    train_sampler = RandomSampler(d_train, replacement=True, num_samples=args.samples_per_epoch) \
                    if args.samples_per_epoch and args.samples_per_epoch>0 else None
    dl_train=DataLoader(d_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
                        sampler=train_sampler, **common)
    dl_val =DataLoader(d_val,  batch_size=args.batch_size, shuffle=False, **common)
    dl_test=DataLoader(d_test, batch_size=args.batch_size, shuffle=False, **common)

    # model & opt
    model=P2CTransformer(
        args.time_window, ny=ny, nx_target=nx,
        d_model=args.d_model, nhead=args.nhead,
        enc_layers=args.num_enc_layers, dec_layers=args.num_dec_layers,
        ffn=args.ffn_dim, drop=args.dropout, act=args.act, out_c=nc
    ).to(args.device)
    if args.compile and torch.__version__ >= "2.0.0":
        model = torch.compile(model, mode="max-autotune")
    optim=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # wandb
    wb = init_wandb(args, model)

    # AMP
    amp_on = args.amp and args.device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on and (amp_dtype==torch.float16))

    # resume / eval
    start_epoch=0; best_val=None
    if args.resume:
        start_epoch,best_val=load_ckpt(args.resume, model, optim, map_location=args.device)

    if args.eval_only:
        vm,va=evaluate(model, dl_val,  args.device, tag="Val",  amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, ny=ny, nx=nx, nc=nc)
        tm,ta=evaluate(model, dl_test, args.device, tag="Test", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, ny=ny, nx=nx, nc=nc)
        print(f"[EvalOnly] Val MSE={vm:.6e} MAE={va:.6e} | Test MSE={tm:.6e} MAE={ta:.6e}")
        if wb is not None: wb.log({"val/MSE": vm, "val/MAE": va, "test/MSE": tm, "test/MAE": ta})
        return

    epoch_bar=tqdm(range(start_epoch, args.epochs), desc="Epochs", position=0)
    for epoch in epoch_bar:
        tr_loss = train_one_epoch(model, dl_train, optim, args.device, epoch, args.epochs, scaler, amp_dtype, amp_on, wb)
        vm, va = evaluate(model, dl_val, args.device, tag="Val", epoch=epoch, epochs=args.epochs,
                          amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, ny=ny, nx=nx, nc=nc)
        epoch_bar.set_postfix(train_loss=f"{tr_loss:.4e}", val_MSE_norm=f"{vm:.4e}", val_MAE_norm=f"{va:.4e}")
        if wb is not None: wb.log({"epoch/train_loss": tr_loss, "epoch/val_MSE": vm, "epoch/val_MAE": va, "epoch": epoch}, commit=True)

        # ckpt
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_last = os.path.join(args.save_dir,"checkpoint_last.pt")
        ckpt_best = os.path.join(args.save_dir,"checkpoint_best.pt")
        save_ckpt(ckpt_last, model, optim, epoch, args, best=best_val, last_only=True)
        if best_val is None or vm<best_val:
            best_val=vm; save_ckpt(ckpt_best, model, optim, epoch, args, best=best_val, last_only=False)
            if wb is not None: wb.log({"epoch/best_val_MSE": best_val, "epoch": epoch}, commit=True)

    tm, ta = evaluate(model, dl_test, args.device, tag="Test", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, ny=ny, nx=nx, nc=nc)
    print(f"[Final] Test MSE={tm:.6e} MAE={ta:.6e}")
    if wb is not None: wb.log({"final/test_MSE": tm, "final/test_MAE": ta})

if __name__=="__main__":
    main()
