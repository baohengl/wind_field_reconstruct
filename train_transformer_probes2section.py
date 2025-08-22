#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, argparse
from typing import List, Tuple
import h5py, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from tqdm.auto import tqdm

# ===================== Utils =====================
def set_seed(seed:int, strict_det:bool=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if strict_det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def split_indices(n:int, tr:float, vr:float, seed:int)->Tuple[List[int],List[int],List[int]]:
    idx=list(range(n)); rng=random.Random(seed); rng.shuffle(idx)
    ntr=int(n*tr); nvr=int(n*vr)
    return idx[:ntr], idx[ntr:ntr+nvr], idx[ntr+nvr:]

# ============ Persistent H5 helpers ============
def _attach_h5_recursive(ds):
    """Attach a persistent h5 file handle to P2SDataset inside (possibly nested) Subset."""
    if isinstance(ds, Subset):
        _attach_h5_recursive(ds.dataset)
    elif isinstance(ds, P2SDataset):
        if getattr(ds, "_h5", None) is None:
            ds._h5 = h5py.File(ds.h5, "r", swmr=True, libver="latest")

def _worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    _attach_h5_recursive(info.dataset)

# ===================== Dataset =====================
class P2SDataset(Dataset):
    def __init__(self, h5:str, time_window:int=21, z_start_1based:int=21,
                 augment:bool=False, time_stride:int=1):
        assert time_window%2==1, "time_window must be odd"
        assert time_stride>=1, "time_stride must be >=1"
        self.h5, self.tw, self.half = h5, time_window, time_window//2
        self.z0, self.augment, self.tstride = max(0,z_start_1based-1), augment, time_stride
        self.items=[]  # (case,z,t)
        self._h5=None  # will be set per-worker

        with h5py.File(h5,"r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 missing /cases")
            for cname,g in f["cases"].items():
                if not isinstance(g,h5py.Group) or ("section" not in g or "probes" not in g): continue
                Ts,Zs,Ys,Xs,Cs=g["section"].shape
                Tp,Zp,P9,Cp=g["probes"].shape
                assert (Ts==Tp and Zs==Zp) and (Ys,Xs,Cs)==(49,41,3) and (P9,Cp)==(9,3), f"{cname} shape mismatch"
                z1, z2 = self.z0, Zs-1
                t1, t2 = self.half, Ts-1-self.half
                if z1>z2 or t1>t2: continue
                for z in range(z1,z2+1):
                    for t in range(t1, t2+1, self.tstride):
                        self.items.append((cname,z,t))
        if not self.items: raise RuntimeError("No valid (case,z,t); check time_window/z_start/time_stride.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname,z,t=self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5,"r")
        g=f["cases"][cname]
        p=g["probes"][t-self.half:t+self.half+1, z, :, :].astype(np.float32) # (T,9,3)
        s=g["section"][t, z, :, :, :].astype(np.float32)                     # (49,41,3)

        if self.augment:
            if random.random()<0.5: s = s[:, ::-1, :]               # flip x(41)
            if random.random()<0.5: s = s[::-1, :, :]; p = p[:, ::-1, :]  # flip y(49) & probes 9

        p = np.ascontiguousarray(p); s = np.ascontiguousarray(s)    # remove negative strides for torch.from_numpy
        p=torch.from_numpy(p.reshape(p.shape[0]*p.shape[1], 3))     # (T*9,3)
        s=torch.from_numpy(s)                                       # (49,41,3)
        if self._h5 is None: f.close()  # only main-thread path
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
        if d_model%2==1: pe[:,1::2]=torch.cos(pos*div[:-1])
        else:            pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(1))
    def forward(self,x):  # x: (L,N,D)
        return x + self.pe[:x.size(0)]

class P2STransformer(nn.Module):
    def __init__(self,time_window:int,d_model:int=192,nhead:int=6,enc_layers:int=3,dec_layers:int=1,ffn:int=768,drop:float=0.1,act:str="gelu"):
        super().__init__()
        self.seq_len=time_window*9; self.out_y,self.out_x,self.out_c=49,41,3; nq=self.out_y*self.out_x
        self.in_proj=nn.Linear(3,d_model); self.pos=PosEnc(d_model, max_len=self.seq_len+8)
        enc = nn.TransformerEncoderLayer(d_model,nhead,ffn,drop,activation=act,batch_first=False,norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model,nhead,ffn,drop,activation=act,batch_first=False,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,enc_layers)
        self.decoder=nn.TransformerDecoder(dec,dec_layers)
        self.q = nn.Parameter(torch.randn(nq,d_model))
        self.qpos = nn.Parameter(torch.zeros(nq,d_model)); nn.init.normal_(self.qpos, std=0.02)
        self.head=nn.Linear(d_model,self.out_c)
    def forward(self,x):  # x: (B,L,3)
        B,L,C=x.shape; assert L==self.seq_len and C==3
        h=self.in_proj(x).transpose(0,1)         # (L,B,D)
        h=self.encoder(self.pos(h))              # (L,B,D)
        q=(self.q+self.qpos).unsqueeze(1).expand(-1,B,-1)  # (nq,B,D)
        y=self.decoder(q,h).transpose(0,1)       # (B,nq,D)
        y=self.head(y).view(B,self.out_y,self.out_x,self.out_c)
        return y

# ================= Train / Eval =================
def enable_flash_attention():
    try:
        # PyTorch<2.3 API；新版本会有弃用警告，但仍可用
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception:
        pass

def train_one_epoch(model, loader, optim, device, epoch:int, epochs:int, scaler, amp_dtype, amp_on:bool, wb):
    model.train(); mse=nn.MSELoss(); running=0.0; n=0
    pbar=tqdm(loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for step, batch in enumerate(pbar):
        x=batch["probes_seq"].to(device, non_blocking=True)
        y=batch["target"].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if amp_on:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                yhat=model(x); loss=mse(yhat,y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
        else:
            yhat=model(x); loss=mse(yhat,y)
            loss.backward(); optim.step()

        b=x.size(0); running+=loss.item()*b; n+=b
        avg = running/max(1,n)
        pbar.set_postfix(loss=f"{avg:.4e}")
        # wandb: per-step logging
        if wb is not None:
            wb.log({"train/loss": loss.item(), "train/avg_loss": avg,
                    "train/step_samples": n, "epoch": epoch}, commit=True)
    return running/max(1,n)

@torch.no_grad()
def evaluate(model, loader, device, tag="Val", epoch:int=None, epochs:int=None, amp_dtype=None, amp_on:bool=False, wb=None):
    model.eval(); mse=nn.MSELoss(reduction="sum"); mae=nn.L1Loss(reduction="sum")
    smse=smae=0.0; n=0
    desc = f"{tag}" if epoch is None else f"{tag} {epoch+1}/{epochs}"
    pbar=tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x=batch["probes_seq"].to(device, non_blocking=True)
        y=batch["target"].to(device, non_blocking=True)
        if amp_on:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                yhat=model(x)
        else:
            yhat=model(x)
        smse+=nn.functional.mse_loss(yhat,y,reduction="sum").item()
        smae+=nn.functional.l1_loss(yhat,y,reduction="sum").item()
        n+=x.size(0)
        denom=max(1, n*49*41*3)
        pbar.set_postfix(MSE=f"{(smse/denom):.4e}", MAE=f"{(smae/denom):.4e}")
    denom=max(1, n*49*41*3)
    m=smse/denom; a=smae/denom
    if wb is not None and epoch is not None:
        wb.log({f"{tag.lower()}/MSE": m, f"{tag.lower()}/MAE": a, "epoch": epoch}, commit=True)
    return m, a

def save_ckpt(path, model, optim, epoch, args, best=None, last_only=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":optim.state_dict(),
                "epoch":epoch,"args":vars(args),"best_val":best}, path)
    if not last_only:
        torch.save(model.state_dict(), os.path.join(os.path.dirname(path),"model_only.pt"))

def load_ckpt(path, model, optim=None, map_location="cpu"):
    ckpt=torch.load(path,map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optim is not None and "optimizer" in ckpt: optim.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch",0)+1, ckpt.get("best_val",None)

# ================== wandb utils ==================
def init_wandb(args, model):
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] Not installed or failed to import: {e}\n"
              f"         Run `pip install wandb` or use --no-wandb to disable.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name if args.wandb_run_name else None,
        mode=args.wandb_mode,            # 'online' | 'offline' | 'disabled'
        config=vars(args),
        resume="allow",
        save_code=True,
    )
    try:
        wandb.watch(model, log="gradients", log_freq=args.wandb_watch_freq, log_graph=False)
    except Exception:
        pass
    return wandb

# ===================== Main =====================
def parse_args():
    p=argparse.ArgumentParser("Transformer: probes(time-window) -> section(t)")
    p.add_argument("--h5", required=True); p.add_argument("--save-dir", default="./runs/probes2section")
    p.add_argument("--resume", default=""); p.add_argument("--eval-only", action="store_true")

    # dataset / sampling
    p.add_argument("--time-window", type=int, default=21)
    p.add_argument("--z-start-1based", type=int, default=21)
    p.add_argument("--time-stride", type=int, default=1, help="sample every k-th time step when building indices")
    p.add_argument("--train-ratio", type=float, default=0.8); p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--samples-per-epoch", type=int, default=0, help="if >0, RandomSampler with this many samples/epoch")
    p.add_argument("--val-samples", type=int, default=0, help="if >0, sample this many from val set per eval")
    p.add_argument("--test-samples", type=int, default=0, help="if >0, sample this many from test set per eval")

    # run
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)

    # optim
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--weight-decay", type=float, default=1e-2)

    # model
    p.add_argument("--d-model", type=int, default=192); p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--num-enc-layers", type=int, default=3); p.add_argument("--num-dec-layers", type=int, default=1)
    p.add_argument("--ffn-dim", type=int, default=768); p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    # perf toggles
    p.add_argument("--amp", action="store_true", help="enable autocast + GradScaler")
    p.add_argument("--bf16", action="store_true", help="use bfloat16 in autocast (H100 recommended)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--cudnn-benchmark", action="store_true", help="speed over strict determinism")

    # device & GPU selection
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="device string, e.g., 'cuda', 'cuda:0', 'cpu'")
    p.add_argument("--gpu", type=int, default=-1,
                   help="choose which GPU index to use (e.g., 0/1/2...). If <0, follow --device")

    # wandb
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="probes2section")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online","offline","disabled"])
    p.add_argument("--wandb-watch-freq", type=int, default=200, help="log gradients every N optimizer steps")
    return p.parse_args()

def maybe_subsample_subset(subset_obj, k:int, seed:int):
    if k<=0 or k>=len(subset_obj): return subset_obj
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(subset_obj), size=k, replace=False)
    pick = sorted(pick.tolist())
    new_idx = [subset_obj.indices[i] for i in pick]
    return Subset(subset_obj.dataset, new_idx)

def resolve_device(args):
    # If user explicitly chose a GPU index
    if args.gpu is not None and args.gpu >= 0:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if args.gpu < n:
                dev = f"cuda:{args.gpu}"
                print(f"[Device] Using user-specified GPU index: {dev}")
                return dev
            else:
                print(f"[Warning] Requested GPU {args.gpu} not available (only {n} found). "
                      f"Falling back to cuda:0." if n > 0 else "[Warning] No CUDA device found. Falling back to CPU.")
                return "cuda:0" if n > 0 else "cpu"
        else:
            print("[Warning] CUDA not available. Falling back to CPU.")
            return "cpu"
    # Otherwise follow --device
    dev = args.device
    if isinstance(dev, str) and dev.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] --device set to CUDA but CUDA not available. Falling back to CPU.")
        dev = "cpu"
    print(f"[Device] Using {dev}")
    return dev

def main():
    args=parse_args()
    set_seed(args.seed, strict_det=not args.cudnn_benchmark)
    torch.set_float32_matmul_precision("high")

    # resolve device from --gpu / --device
    args.device = resolve_device(args)

    if str(args.device).startswith("cuda"):
        enable_flash_attention()

    # wandb init
    wb = init_wandb(args, model=None)  # model 未就绪，先让 config 同步

    # datasets
    d_train_full=P2SDataset(args.h5, args.time_window, args.z_start_1based, augment=True,  time_stride=args.time_stride)
    N=len(d_train_full); tr_idx, va_idx, te_idx = split_indices(N, args.train_ratio, args.val_ratio, args.seed)
    d_train=Subset(d_train_full, tr_idx)

    d_val_full = P2SDataset(args.h5, args.time_window, args.z_start_1based, augment=False, time_stride=args.time_stride)
    d_val  = Subset(d_val_full,  va_idx)
    d_test = Subset(d_val_full,  te_idx)
    d_val  = maybe_subsample_subset(d_val,  args.val_samples,  args.seed+1)
    d_test = maybe_subsample_subset(d_test, args.test_samples, args.seed+2)

    # samplers / loaders
    train_sampler = None
    if args.samples_per_epoch and args.samples_per_epoch>0:
        train_sampler = RandomSampler(d_train, replacement=True, num_samples=args.samples_per_epoch)

    common_dl_args = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
                          persistent_workers=(args.num_workers>0), prefetch_factor=args.prefetch_factor,
                          worker_init_fn=_worker_init_fn)

    dl_train=DataLoader(d_train, batch_size=args.batch_size,
                        shuffle=(train_sampler is None), sampler=train_sampler, **common_dl_args)
    dl_val  =DataLoader(d_val,   batch_size=args.batch_size, shuffle=False, **common_dl_args)
    dl_test =DataLoader(d_test,  batch_size=args.batch_size, shuffle=False, **common_dl_args)

    # model & opt
    model=P2STransformer(args.time_window,args.d_model,args.nhead,args.num_enc_layers,args.num_dec_layers,
                         args.ffn_dim,args.dropout,args.act).to(args.device)
    if args.compile and torch.__version__ >= "2.0.0":
        model = torch.compile(model, mode="max-autotune")

    optim=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # wandb: now watch model if enabled
    if wb is not None:
        try:
            import wandb
            wandb.watch(model, log="gradients", log_freq=args.wandb_watch_freq, log_graph=False)
        except Exception:
            pass

    start_epoch=0; best_val=None
    if args.resume:
        start_epoch,best_val=load_ckpt(args.resume, model, optim, map_location=args.device)

    # AMP/bf16
    amp_on = args.amp and str(args.device).startswith("cuda")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on and (amp_dtype==torch.float16))  # bfloat16 不需要 scaler

    if args.eval_only:
        vm,va=evaluate(model, dl_val, args.device, tag="Val", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb)
        tm,ta=evaluate(model, dl_test, args.device, tag="Test", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb)
        print(f"[EvalOnly] Val MSE={vm:.6e} MAE={va:.6e} | Test MSE={tm:.6e} MAE={ta:.6e}")
        if wb is not None:
            wb.log({"val/MSE": vm, "val/MAE": va, "test/MSE": tm, "test/MAE": ta})
        return

    epoch_bar=tqdm(range(start_epoch, args.epochs), desc="Epochs", position=0)
    for epoch in epoch_bar:
        tr_loss = train_one_epoch(model, dl_train, optim, args.device, epoch, args.epochs, scaler, amp_dtype, amp_on, wb)
        vm, va = evaluate(model, dl_val, args.device, tag="Val", epoch=epoch, epochs=args.epochs,
                          amp_dtype=amp_dtype, amp_on=amp_on, wb=wb)
        epoch_bar.set_postfix(train_loss=f"{tr_loss:.4e}", val_MSE=f"{vm:.4e}", val_MAE=f"{va:.4e}")

        # wandb: epoch-level metrics
        if wb is not None:
            wb.log({"epoch/train_loss": tr_loss, "epoch/val_MSE": vm, "epoch/val_MAE": va, "epoch": epoch}, commit=True)

        # checkpoints
        save_ckpt(os.path.join(args.save_dir,"checkpoint_last.pt"), model, optim, epoch, args, best=best_val, last_only=True)
        if best_val is None or vm<best_val:
            best_val=vm
            save_ckpt(os.path.join(args.save_dir,"checkpoint_best.pt"), model, optim, epoch, args, best=best_val, last_only=False)
            if wb is not None:
                wb.log({"epoch/best_val_MSE": best_val, "epoch": epoch}, commit=True)

    tm, ta = evaluate(model, dl_test, args.device, tag="Test", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb)
    print(f"[Final] Test MSE={tm:.6e} MAE={ta:.6e}")
    if wb is not None:
        wb.log({"final/test_MSE": tm, "final/test_MAE": ta})

if __name__=="__main__":
    main()
