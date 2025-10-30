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
    """Attach a persistent h5 file handle to dataset inside (possibly nested) Subset."""
    while isinstance(ds, Subset):
        ds = ds.dataset
    if isinstance(ds, P2CBandDataset) and getattr(ds, "_h5", None) is None:
        ds._h5 = h5py.File(ds.h5, "r", swmr=True, libver="latest")

def _worker_init_fn(_worker_id):
    info = torch.utils.data.get_worker_info()
    _attach_h5_recursive(info.dataset)

# ===================== Dataset =====================
class P2CBandDataset(Dataset):
    """
    Probes time-window (ending at 'step') -> Center-x column over a z-band at time 'step'.

    Input  (X): probes_seq at fixed z over [step - tlength + 1, ..., step], shape (tlength, 9, 3) -> flatten (tlength*9, 3)
    Target (Y): section[time=step, z:z+zlength, y=0..Ny-1, x=x_center, comps=target_comps], shape (zlength, Ny, C_out)

    Normalization:
      - If present, use /meta/section_velocity_mean (3,) and /meta/section_velocity_std (3,) for both probes and targets
      - Target comps subset accordingly (e.g., 'yz' -> indices [1,2])
    """
    def __init__(
        self,
        h5:str,
        tlength:int=81,
        zlength:int=5,
        time_stride:int=1,
        augment:bool=False,
        normalize:bool=True,
        eps:float=1e-6,
        target_comp_idx: Optional[List[int]] = None,
        x_center_1based: Optional[int] = None,  # if None, use center column by shape
    ):
        assert tlength >= 1 and zlength >= 1 and time_stride >= 1
        self.h5 = h5
        self.tlength = int(tlength)
        self.zlength = int(zlength)
        self.tstride = int(time_stride)
        self.augment = bool(augment)
        self.items=[]; self._h5=None

        # target channels
        t_idx = [1,2] if not target_comp_idx else sorted(set(int(i) for i in target_comp_idx))
        if any(i not in (0,1,2) for i in t_idx): raise ValueError("target_comp_idx 仅允许 0/1/2（对应 x/y/z）")
        self.target_comp_idx = t_idx

        # normalization stats
        self.normalize = normalize
        self.eps = float(eps)
        self.norm_mean = np.zeros(3, dtype=np.float32)
        self.norm_std  = np.ones(3, dtype=np.float32)
        try:
            with h5py.File(h5, "r") as f:
                if normalize and "meta" in f and \
                   "section_velocity_mean" in f["meta"] and "section_velocity_std" in f["meta"]:
                    m = f["meta"]["section_velocity_mean"][...].astype(np.float32).reshape(3)
                    s = np.maximum(f["meta"]["section_velocity_std"][...].astype(np.float32).reshape(3), self.eps)
                    self.norm_mean[:] = m; self.norm_std[:] = s
                else:
                    if normalize: print("[Normalize] 未找到 meta/{section_velocity_mean,std}，禁用归一化。")
                    self.normalize = False
        except Exception as e:
            if normalize: print(f"[Normalize] 读取 meta 失败（{e}），禁用归一化。")
            self.normalize = False

        # build index list across cases
        with h5py.File(h5, "r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 缺少 /cases")
            for cname, g in f["cases"].items():
                if not isinstance(g, h5py.Group) or ("section" not in g or "probes" not in g):
                    continue
                Ts, Zs, Ny, Xs, Cs = g["section"].shape    # expected Cs == 3
                Tp, Zp, P9, Cp    = g["probes"].shape     # expected P9 == 9, Cp == 3
                if not (Ts==Tp and Zs==Zp and Cs==3 and Cp==3 and P9==9):
                    raise RuntimeError(f"{cname}: shape mismatch - section {g['section'].shape}, probes {g['probes'].shape}")

                # center x column (0-based)
                if x_center_1based is None:
                    xc = Xs // 2
                else:
                    xc = int(x_center_1based) - 1
                if not (0 <= xc < Xs):
                    raise RuntimeError(f"{cname}: x_center 索引无效 (xc={xc}, Xs={Xs})")
                self.xc = xc
                self._ny = Ny

                # valid ranges to avoid OOB
                t_min = self.tlength - 1
                t_max = Ts - 1
                z_min = 0
                z_max = Zs - self.zlength

                if t_min > t_max or z_min > z_max:
                    continue

                for z in range(z_min, z_max+1):
                    for step in range(t_min, t_max+1, self.tstride):
                        self.items.append((cname, z, step))

        if not self.items:
            raise RuntimeError("没有可用样本（检查 tlength、zlength、time_stride 是否过大）。")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname, z, step = self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5, "r")
        g = f["cases"][cname]

        # X: probes over [step - tlength + 1, ..., step] at fixed z
        t0 = step - self.tlength + 1
        t1 = step
        probes = g["probes"][t0:t1+1, z, :, :].astype(np.float32)  # (tlength, 9, 3)

        # Y: section at time=step, z-band, center x, target comps
        # section: (T, Z, Ny, X, 3)
        sec = g["section"][step, z:z+self.zlength, :, self.xc, :].astype(np.float32)  # (zlength, Ny, 3)
        target = sec[..., self.target_comp_idx]                                       # (zlength, Ny, C_out)

        # simple augmentation (optional & safe)
        if self.augment:
            if random.random() < 0.5:
                target = target[:, ::-1, :]  # flip y

        if self.normalize:
            probes = (probes - self.norm_mean) / self.norm_std
            target = (target - self.norm_mean[self.target_comp_idx]) / np.maximum(self.norm_std[self.target_comp_idx], self.eps)

        x = torch.from_numpy(np.ascontiguousarray(probes.reshape(-1, 3)))  # (tlength*9, 3)
        y = torch.from_numpy(np.ascontiguousarray(target))                 # (zlength, Ny, C_out)

        if self._h5 is None: f.close()
        return {"probes_seq": x, "target": y, "meta": {"case": cname, "z": z, "step": step}}

def collate(batch):
    x=torch.stack([b["probes_seq"] for b in batch],0)  # (B, L, 3)
    y=torch.stack([b["target"]     for b in batch],0)  # (B, zlength, Ny, C)
    return {"probes_seq":x, "target":y, "meta":[b["meta"] for b in batch]}

# ===================== Model =====================
class PosEnc(nn.Module):
    def __init__(self,d_model:int,max_len:int=20032):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div[:-1] if d_model%2==1 else pos*div)
        self.register_buffer("pe",pe.unsqueeze(1))
    def forward(self,x):  # (L,N,D)
        return x + self.pe[:x.size(0)]

class PosEnc2D(nn.Module):
    """
    可区分 z 与 y 两个方向的二维正弦位置编码：
    对每个 token(z,y) 生成长度 D 的编码，方式为将两套 1D 正弦编码（z 与 y）拼合/交错后映射到 D 维。
    这里实现为：分别用一半维度编码 z，一半维度编码 y（若 D 为奇数则 y 少 1）。
    """
    def __init__(self, d_model:int, zlength:int, ny:int):
        super().__init__()
        self.d_model = d_model
        dz = d_model // 2
        dy = d_model - dz
        self.register_buffer("pe2d", self._build_2d_sincos(dz, dy, zlength, ny), persistent=False)  # (Z*Y, D)

    @staticmethod
    def _build_1d_sincos(L:int, dim:int):
        pos = torch.arange(L).float().unsqueeze(1)  # (L,1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(L, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div)[:, :-1]  # 对齐长度
        return pe  # (L, dim)

    def _build_2d_sincos(self, dz:int, dy:int, zlength:int, ny:int):
        pez = self._build_1d_sincos(zlength, dz)  # (Z, dz)
        pey = self._build_1d_sincos(ny, dy)       # (Y, dy)
        # 组合：每个(z,y)的编码 = concat(pez[z], pey[y])
        pe2d = torch.zeros(zlength, ny, dz + dy)
        for z in range(zlength):
            pe2d[z, :, :dz] = pez[z].unsqueeze(0).expand(ny, -1)
            pe2d[z, :, dz:] = pey
        return pe2d.reshape(zlength*ny, dz+dy)  # (Z*Y, D)

    def forward(self, tokens):  # tokens: (B, Z*Y, D)
        # 直接相加：广播到 batch
        return tokens + self.pe2d.unsqueeze(0).to(tokens.dtype).to(tokens.device)

class SpatialRefiner2D(nn.Module):
    """
    2D Transformer 空间精炼模块：
    - 输入：decoder 的特征 (B, Z*Y, D)
    - 位置：添加 2D 位置编码后，送入 TransformerEncoder 做自注意力
    - 输出：同形状 (B, Z*Y, D)
    """
    def __init__(self, d_model:int, zlength:int, ny:int,
                 nhead:int=4, num_layers:int=2, ffn:int=512, drop:float=0.1, act:str="gelu"):
        super().__init__()
        enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_ly, num_layers)
        self.pos2d   = PosEnc2D(d_model, zlength, ny)

    def forward(self, y_tokens):  # (B, Z*Y, D)
        y_tokens = self.pos2d(y_tokens)
        y_tokens = self.encoder(y_tokens)
        return y_tokens

class P2CBandTransformer(nn.Module):
    """
    Encoder over probe time sequence; decoder queries (zlength*Ny) tokens -> (zlength, Ny, C_out)
    可选：在 decoder 输出后添加 2D Transformer 精炼模块（方案二）
    """
    def __init__(self, tlength:int, ny:int, zlength:int,
                 d_model:int=256, nhead:int=8, enc_layers:int=6, dec_layers:int=2,
                 ffn:int=1024, drop:float=0.1, act:str="gelu", out_c:int=2,
                 use_spatial_refine:bool=False,
                 refine_nhead:int=4, refine_layers:int=2, refine_ffn:int=512, refine_drop:float=0.1, refine_act:str="gelu"):
        super().__init__()
        self.seq_len = tlength * 9
        self.out_zy = zlength * ny
        self.zlength, self.ny, self.out_c = zlength, ny, out_c
        self.use_spatial_refine = bool(use_spatial_refine)

        self.in_proj = nn.Linear(3, d_model)
        self.pos = PosEnc(d_model, max_len=self.seq_len+8)

        enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=False, norm_first=True)
        dec_ly = nn.TransformerDecoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=False, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_ly, enc_layers)
        self.decoder = nn.TransformerDecoder(dec_ly, dec_layers)

        # learnable queries for (zlength * Ny)
        self.q = nn.Parameter(torch.randn(self.out_zy, d_model))
        self.qpos = nn.Parameter(torch.zeros(self.out_zy, d_model)); nn.init.normal_(self.qpos, std=0.02)

        # 可选的 2D 空间精炼模块（放在 head 之前）
        if self.use_spatial_refine:
            self.srefine = SpatialRefiner2D(
                d_model=d_model, zlength=zlength, ny=ny,
                nhead=refine_nhead, num_layers=refine_layers, ffn=refine_ffn,
                drop=refine_drop, act=refine_act
            )
        else:
            self.srefine = None

        self.head = nn.Linear(d_model, out_c)

    def forward(self, x):  # x: (B, L, 3)
        B, L, C = x.shape
        assert L == self.seq_len and C == 3
        # 时间编码器
        h = self.in_proj(x).transpose(0,1)     # (L,B,D)
        h = self.encoder(self.pos(h))          # (L,B,D)
        # 解码器：为 (zlength*Ny) 个 query 取特征
        q = (self.q + self.qpos).unsqueeze(1).expand(-1, B, -1)  # (out_zy, B, D)
        y = self.decoder(q, h).transpose(0,1)  # (B, out_zy, D)

        # 2D 空间精炼（可选）
        if self.srefine is not None:
            y = self.srefine(y)               # (B, out_zy, D)

        y = self.head(y)                       # (B, out_zy, C_out)
        return y.view(B, self.zlength, self.ny, self.out_c)

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
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on \
              else torch.cpu.amp.autocast(enabled=False)
        with ctx:
            yhat = model(x)
            loss = mse(yhat, y)
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
             amp_dtype=None, amp_on:bool=False, wb=None, zlength:int=5, ny:int=49, nc:int=2):
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
        denom = max(1, n * zlength * ny * nc)
        pbar.set_postfix(MSE=f"{(smse/denom):.4e}", MAE=f"{(smae/denom):.4e}")
    denom = max(1, n * zlength * ny * nc)
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
    s=(s or "yz").lower().strip()
    idx=sorted(set(m[c] for c in s if c in m))
    if not idx: raise ValueError(f"--target-comps '{s}' 无有效通道，示例：y / z / yz / xyz")
    return idx

def parse_args():
    p=argparse.ArgumentParser("Transformer: probes(time-window ending at step) -> center-x z-band at step")
    # paths
    p.add_argument("--h5", type=str,
                   default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_interpolation_all_cases.h5")
    p.add_argument("--save-dir", type=str, default="./runs/probes2centerband")

    # mode
    p.add_argument("--resume", default="")
    p.add_argument("--eval-only", action="store_true")

    # dataset / sampling
    p.add_argument("--tlength", type=int, default=81, help="输入时间窗口长度（覆盖 [step-tlength+1, step]）")
    p.add_argument("--zlength", type=int, default=5, help="预测的 z 带宽（z..z+zlength-1）")
    p.add_argument("--time-stride", type=int, default=1, help="样本间 step 的步距")
    p.add_argument("--x-center-1based", type=int, default=None, help="中心 x 列（1-based），不设则自动取中间列")
    p.add_argument("--augment", action="store_true")

    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)

    p.add_argument("--samples-per-epoch", type=int, default=0, help=">0 则训练集采样器按此数量重复采样")
    p.add_argument("--val-samples", type=int, default=0)
    p.add_argument("--test-samples", type=int, default=0)

    # normalization
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--target-comps", type=str, default="yz", help="目标分量，默认 yz，可选 y/z/yz/xyz 等")

    # run
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)

    # optim
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    # model (main)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-enc-layers", type=int, default=6)
    p.add_argument("--num-dec-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    # 2D spatial refinement (方案二)
    p.add_argument("--use-2d-refine", action="store_true", help="启用 2D Transformer 空间精炼（decoder 后）")
    p.add_argument("--refine-layers", type=int, default=2, help="2D 精炼的 encoder 层数")
    p.add_argument("--refine-nhead", type=int, default=4, help="2D 精炼的注意力头数")
    p.add_argument("--refine-ffn-dim", type=int, default=512, help="2D 精炼的前馈维度")
    p.add_argument("--refine-dropout", type=float, default=0.1, help="2D 精炼的 dropout")
    p.add_argument("--refine-act", type=str, default="gelu", choices=["relu","gelu"], help="2D 精炼的激活函数")

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
    p.add_argument("--wandb-project", type=str, default="probes2centerband")
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

def _clone_split_with_stats(base: P2CBandDataset, idx: List[int], normalize_flag: bool):
    """Clone dataset with same stats and restrict to idx (Subset)."""
    ds = P2CBandDataset(
        base.h5, tlength=base.tlength, zlength=base.zlength, time_stride=base.tstride,
        augment=False, normalize=normalize_flag, target_comp_idx=base.target_comp_idx,
        x_center_1based=(base.xc+1)
    )
    # reuse mean/std
    ds.norm_mean[:] = base.norm_mean
    ds.norm_std[:]  = np.maximum(base.norm_std, 1e-6)
    return Subset(ds, idx)

def main():
    args = parse_args()
    target_idx = _parse_target_comps(args.target_comps)

    set_seed(args.seed, strict_det=not args.cudnn_benchmark)
    torch.set_float32_matmul_precision("high")

    # device
    args.device = resolve_device(args)
    if args.device.startswith("cuda"): enable_flash_attention()

    # datasets (train/full)
    d_train_full = P2CBandDataset(
        args.h5, tlength=args.tlength, zlength=args.zlength, time_stride=args.time_stride,
        augment=args.augment, normalize=(not args.no_normalize),
        target_comp_idx=target_idx, x_center_1based=args.x_center_1based
    )
    zlen, ny, nc = d_train_full.zlength, d_train_full._ny, len(target_idx)

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
    model = P2CBandTransformer(
        tlength=args.tlength, ny=ny, zlength=zlen,
        d_model=args.d_model, nhead=args.nhead,
        enc_layers=args.num_enc_layers, dec_layers=args.num_dec_layers,
        ffn=args.ffn_dim, drop=args.dropout, act=args.act, out_c=nc,
        use_spatial_refine=args.use_2d_refine,
        refine_nhead=args.refine_nhead, refine_layers=args.refine_layers,
        refine_ffn=args.refine_ffn_dim, refine_drop=args.refine_dropout, refine_act=args.refine_act
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
        vm,va=evaluate(model, dl_val,  args.device, tag="Val",  amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, zlength=zlen, ny=ny, nc=nc)
        tm,ta=evaluate(model, dl_test, args.device, tag="Test", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, zlength=zlen, ny=ny, nc=nc)
        print(f"[EvalOnly] Val MSE={vm:.6e} MAE={va:.6e} | Test MSE={tm:.6e} MAE={ta:.6e}")
        if wb is not None: wb.log({"val/MSE": vm, "val/MAE": va, "test/MSE": tm, "test/MAE": ta})
        return

    os.makedirs(args.save_dir, exist_ok=True)
    epoch_bar=tqdm(range(start_epoch, args.epochs), desc="Epochs", position=0)
    for epoch in epoch_bar:
        tr_loss = train_one_epoch(model, dl_train, optim, args.device, epoch, args.epochs, scaler, amp_dtype, amp_on, wb)
        vm, va = evaluate(model, dl_val, args.device, tag="Val", epoch=epoch, epochs=args.epochs,
                          amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, zlength=zlen, ny=ny, nc=nc)
        epoch_bar.set_postfix(train_loss=f"{tr_loss:.4e}", val_MSE_norm=f"{vm:.4e}", val_MAE_norm=f"{va:.4e}")
        if wb is not None: wb.log({"epoch/train_loss": tr_loss, "epoch/val_MSE": vm, "epoch/val_MAE": va, "epoch": epoch}, commit=True)

        # ckpt
        ckpt_last = os.path.join(args.save_dir,"checkpoint_last.pt")
        ckpt_best = os.path.join(args.save_dir,"checkpoint_best.pt")
        save_ckpt(ckpt_last, model, optim, epoch, args, best=best_val, last_only=True)
        if best_val is None or vm<best_val:
            best_val=vm; save_ckpt(ckpt_best, model, optim, epoch, args, best=best_val, last_only=False)
            if wb is not None: wb.log({"epoch/best_val_MSE": best_val, "epoch": epoch}, commit=True)

    tm, ta = evaluate(model, dl_test, args.device, tag="Test", amp_dtype=amp_dtype, amp_on=amp_on, wb=wb, zlength=zlen, ny=ny, nc=nc)
    print(f"[Final] Test MSE={tm:.6e} MAE={ta:.6e}")
    if wb is not None: wb.log({"final/test_MSE": tm, "final/test_MAE": ta})

if __name__=="__main__":
    main()
