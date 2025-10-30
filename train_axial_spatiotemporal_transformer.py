#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatiotemporal Transformer (Time-Global + Spatial Axial z→y)
for section history -> future z-band regression.

Dataset slice (strictly as requested)
-------------------------------------
Input  X: section[step-Tlength:step, z-zlength:z, :, x_center, comps] -> (Tlength, zlength, Ny, C)
Target Y: section[step,          z+1:z+zlength, :, x_center, comps]  -> (zlength, Ny, C)

HDF5 expected layout
--------------------
/cases/<case_name>/section : (T, Z, Ny, Nx, 3) float16/float32
/meta/section_velocity_mean: (3,) (optional)
/meta/section_velocity_std : (3,) (optional)

Run (example)
-------------
python train_spatiotemporal_axial.py \
  --h5 /vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_interpolation_all_cases.h5 \
  --save-dir ./runs/timeGlobal_axialZY_v1 \
  --tlength 81 --zlength 5 --batch-size 32 --epochs 30 \
  --x-center-1based 21 --target-comps yz --amp --cudnn-benchmark
"""

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

# ============ H5 helpers ============
def _attach_h5_recursive(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    if isinstance(ds, SectionBandCausalDataset) and getattr(ds, "_h5", None) is None:
        ds._h5 = h5py.File(ds.h5, "r", swmr=True, libver="latest")

def _worker_init_fn(_worker_id):
    info = torch.utils.data.get_worker_info()
    _attach_h5_recursive(info.dataset)

# ===================== Dataset =====================
class SectionBandCausalDataset(Dataset):
    """
    Build samples (case, step, z) with bounds:
      step ∈ [Tlength, T-1]
      z    ∈ [zlength, Z - zlength - 1]

    X  = section[step-Tlength:step, z-zlength:z, :, x_center, comps]   -> (Tlength, zlength, Ny, C)
    Y  = section[step,           z+1:z+zlength, :, x_center, comps]    -> (zlength, Ny, C)
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
        x_center_1based: Optional[int] = 21,
    ):
        assert tlength>=1 and zlength>=1 and time_stride>=1
        self.h5 = h5
        self.tlength = int(tlength)
        self.zlength = int(zlength)
        self.tstride = int(time_stride)
        self.augment = bool(augment)
        self.items=[]; self._h5=None

        # target channels
        t_idx = [1,2] if not target_comp_idx else sorted(set(int(i) for i in target_comp_idx))
        if any(i not in (0,1,2) for i in t_idx):
            raise ValueError("target_comp_idx 仅允许 0/1/2（对应 x/y/z）")
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

        # index across cases
        with h5py.File(h5, "r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 缺少 /cases")
            for cname, g in f["cases"].items():
                if not isinstance(g, h5py.Group) or ("section" not in g):
                    continue
                T, Z, Ny, Nx, C = g["section"].shape
                if C != 3:
                    raise RuntimeError(f"{cname}: section[...,3] 预期为 3 分量，得到 {C}")

                # center x column (0-based)
                xc = (Nx // 2) if x_center_1based is None else int(x_center_1based) - 1
                if not (0 <= xc < Nx):
                    raise RuntimeError(f"{cname}: x_center 无效 (xc={xc}, Nx={Nx})")
                self.xc = xc
                self._ny = Ny

                # valid ranges
                step_min = self.tlength
                step_max = T - 1
                z_min = self.zlength
                z_max = Z - self.zlength - 1

                if step_min > step_max or z_min > z_max:
                    continue

                for z in range(z_min, z_max+1):
                    for step in range(step_min, step_max+1, self.tstride):
                        self.items.append((cname, z, step))

        if not self.items:
            raise RuntimeError("没有可用样本（检查 tlength、zlength、time_stride 是否过大）。")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname, z, step = self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5, "r")
        g = f["cases"][cname]

        t0 = step - self.tlength
        t1 = step  # exclusive
        zh0 = z - self.zlength
        zh1 = z   # exclusive
        zf0 = z + 1
        zf1 = z + 1 + self.zlength

        x = g["section"][t0:t1, zh0:zh1, :, self.xc, :].astype(np.float32)  # (T,Zhist,Ny,3)
        x = x[..., self.target_comp_idx]                                     # (T,Zhist,Ny,C)
        y = g["section"][step, zf0:zf1, :, self.xc, :].astype(np.float32)    # (Zout,Ny,3)
        y = y[..., self.target_comp_idx]                                     # (Zout,Ny,C)

        if self.augment and random.random() < 0.5:
            x = x[:, :, ::-1, :]  # flip y
            y = y[:, ::-1, :]

        if self.normalize:
            mean = self.norm_mean[self.target_comp_idx]
            std  = np.maximum(self.norm_std[self.target_comp_idx], self.eps)
            x = (x - mean) / std
            y = (y - mean) / std

        X = torch.from_numpy(np.ascontiguousarray(x))  # (T, Zhist, Ny, C)
        Y = torch.from_numpy(np.ascontiguousarray(y))  # (Zout, Ny, C)

        if self._h5 is None: f.close()
        return {"hist": X, "target": Y, "meta": {"case": cname, "z": z, "step": step}}

def collate(batch):
    X = torch.stack([b["hist"]   for b in batch],0)  # (B, T, Zhist, Ny, C)
    Y = torch.stack([b["target"] for b in batch],0)  # (B, Zout, Ny, C)
    return {"hist": X, "target": Y, "meta": [b["meta"] for b in batch]}

# ===================== Model =====================
class PosEnc1D(nn.Module):
    def __init__(self, d:int, length:int):
        super().__init__()
        self.register_buffer("pe", self._build(d, length), persistent=False)
    @staticmethod
    def _build(d, L):
        pe = torch.zeros(L, d)
        pos = torch.arange(L).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div[:pe[:,1::2].shape[1]])
        return pe
    def forward(self, x):  # (B,L,D)
        return x + self.pe.unsqueeze(0).to(x.device).to(x.dtype)

class PosEnc2D(nn.Module):
    """2D sinusoidal PE for (z,y)."""
    def __init__(self, d_model:int, zlen:int, ny:int):
        super().__init__()
        dz = d_model // 2
        dy = d_model - dz
        self.register_buffer("pez", PosEnc1D._build(dz, zlen), persistent=False)
        self.register_buffer("pey", PosEnc1D._build(dy, ny),   persistent=False)
        self.zlen, self.ny, self.dz, self.dy = zlen, ny, dz, dy
    def forward(self, x):  # (B, Z*Y, D)
        B, L, D = x.shape
        zlen, ny, dz, dy = self.zlen, self.ny, self.dz, self.dy
        pez = self.pez.to(x.device).to(x.dtype)   # (Z,dz)
        pey = self.pey.to(x.device).to(x.dtype)   # (Y,dy)
        pe = torch.zeros(zlen, ny, D, device=x.device, dtype=x.dtype)
        pe[:,:, :dz] = pez.unsqueeze(1).expand(-1, ny, -1)
        pe[:,:, dz:] = pey.unsqueeze(0).expand(zlen, -1, -1)
        pe = pe.reshape(1, zlen*ny, D)
        return x + pe

def causal_mask(T:int, device):
    """Upper-triangular mask that prevents attending to future time steps."""
    m = torch.full((T, T), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m

class TemporalTransformerEncoder(nn.Module):
    """Time-global Transformer over length-T sequences at each (z,y)."""
    def __init__(self, c_in:int, d_model:int, nhead:int=4, layers:int=2, ffn:int=512, drop:float=0.1, act:str="gelu"):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.pe = None  # 可以加时间位置编码；这里用 causal mask 足够
    def forward(self, x):  # x: (B*Z*Y, T, C_in)
        BZY, T, _ = x.shape
        h = self.proj(x)  # (BZY,T,D)
        mask = causal_mask(T, x.device)
        h = self.encoder(h, mask=mask)  # (BZY,T,D)
        return h[:, -1, :]              # 取最后时刻表征 -> (BZY,D)

class AxialSelfAttentionZY(nn.Module):
    """
    Axial attention on (Z,Y):
      1) along z (for each y line): sequences of length Z, repeated Y times
      2) along y (for each z line): sequences of length Y, repeated Z times
    """
    def __init__(self, d_model:int, nhead:int=4, layers:int=1, ffn:int=512, drop:float=0.1, act:str="gelu"):
        super().__init__()
        enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=True, norm_first=True)
        self.z_block = nn.TransformerEncoder(enc_ly, layers)
        self.y_block = nn.TransformerEncoder(enc_ly, layers)

    def forward(self, mem):  # mem: (B, Z, Y, D)
        B,Z,Y,D = mem.shape
        # z-axis: treat each y column separately -> (B*Y, Z, D)
        h = mem.permute(0,2,1,3).reshape(B*Y, Z, D)
        h = self.z_block(h)
        h = h.reshape(B, Y, Z, D).permute(0,2,1,3)  # (B,Z,Y,D)

        # y-axis: treat each z row separately -> (B*Z, Y, D)
        h2 = h.reshape(B*Z, Y, D)
        h2 = self.y_block(h2)
        h2 = h2.reshape(B, Z, Y, D)
        return h2

class SpatialRefiner2D(nn.Module):
    def __init__(self, d_model:int, zlen:int, ny:int,
                 nhead:int=4, num_layers:int=1, ffn:int=512, drop:float=0.1, act:str="gelu"):
        super().__init__()
        enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_ly, num_layers)
        self.pos2d   = PosEnc2D(d_model, zlen, ny)
    def forward(self, y_tokens):  # (B, Z*Y, D)
        y_tokens = self.pos2d(y_tokens)
        return self.encoder(y_tokens)

class AxialSpatiotemporalTransformer(nn.Module):
    def __init__(self, tlength:int, zlen:int, ny:int, out_c:int=2,
                 d_model:int=256, t_nhead:int=8, t_layers:int=2, t_ffn:int=1024,
                 axial_nhead:int=4, axial_layers:int=1, axial_ffn:int=512,
                 dec_nhead:int=8, dec_layers:int=2, dec_ffn:int=1024,
                 drop:float=0.1, act:str="gelu", use_spatial_refine:bool=False,
                 refine_layers:int=1, refine_nhead:int=4, refine_ffn:int=512, refine_drop:float=0.1, refine_act:str="gelu"):
        super().__init__()
        self.tlength, self.zlen, self.ny, self.out_c = tlength, zlen, ny, out_c
        self.d_model = d_model

        # 1) time-global encoder (causal) per (z,y)
        self.temp_enc = TemporalTransformerEncoder(c_in=out_c, d_model=d_model,
                                                   nhead=t_nhead, layers=t_layers, ffn=t_ffn, drop=drop, act=act)

        # 2) spatial axial attention on memory (z→y)
        self.axial = AxialSelfAttentionZY(d_model=d_model, nhead=axial_nhead, layers=axial_layers,
                                          ffn=axial_ffn, drop=drop, act=act)

        # 3) decoder: future queries (z*Y) cross-attend to memory (z*Y)
        dec_ly = nn.TransformerDecoderLayer(d_model, dec_nhead, dec_ffn, drop, activation=act,
                                            batch_first=False, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_ly, dec_layers)

        # learned queries and 2D pos enc
        self.mem_pos = PosEnc2D(d_model, zlen, ny)  # history window PE
        self.q = nn.Parameter(torch.randn(zlen*ny, d_model))
        nn.init.normal_(self.q, std=0.02)
        self.q_pos = PosEnc2D(d_model, zlen, ny)    # future window PE (Δz,y)

        # optional 2D refine on decoder outputs
        self.use_spatial_refine = bool(use_spatial_refine)
        if self.use_spatial_refine:
            self.srefine = SpatialRefiner2D(d_model, zlen, ny, nhead=refine_nhead,
                                            num_layers=refine_layers, ffn=refine_ffn,
                                            drop=refine_drop, act=refine_act)
        else:
            self.srefine = None

        self.head = nn.Linear(d_model, out_c)

    def forward(self, x):  # x: (B, T, Zhist, Ny, C)
        B,T,Z,Y,C = x.shape
        assert Z == self.zlen and Y == self.ny and C == self.out_c

        # ------ time-global encode each (z,y) ------
        # reshape to (B*Z*Y, T, C)
        h = x.permute(0,2,3,1,4).reshape(B*Z*Y, T, C)
        h = self.temp_enc(h)                    # (B*Z*Y, D)
        h = h.view(B, Z, Y, self.d_model)       # memory grid

        # ------ spatial axial (z→y) on memory ------
        h = self.axial(h)                       # (B, Z, Y, D)

        # ------ build memory tokens + 2D pos ------
        mem = h.reshape(B, Z*Y, self.d_model)
        mem = self.mem_pos(mem)                 # (B, Z*Y, D)
        mem = mem.transpose(0,1)                # (Z*Y, B, D) for decoder

        # ------ future queries (learned + pos) ------
        q = self.q.unsqueeze(0).expand(B, -1, -1)   # (B, Z*Y, D)
        q = self.q_pos(q)
        q = q.transpose(0,1)                        # (Z*Y, B, D)

        # ------ cross-attend ------
        y = self.decoder(q, mem).transpose(0,1)     # (B, Z*Y, D)

        if self.use_spatial_refine:
            y = self.srefine(y)                     # (B, Z*Y, D)

        y = self.head(y).view(B, self.zlen, self.ny, self.out_c)  # (B, Zout, Ny, C)
        return y

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
        X=batch["hist"].to(device, non_blocking=True)
        Y=batch["target"].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else torch.cpu.amp.autocast(enabled=False)
        with ctx:
            Yhat = model(X)
            loss = mse(Yhat, Y)
        if amp_on:
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        else:
            loss.backward(); optim.step()
        b=X.size(0); running+=loss.item()*b; n+=b
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
        X = batch["hist"].to(device, non_blocking=True)
        Y = batch["target"].to(device, non_blocking=True)
        with ctx:
            Yhat = model(X)
        smse += F.mse_loss(Yhat, Y, reduction="sum").item()
        smae += F.l1_loss (Yhat, Y, reduction="sum").item()
        n += X.size(0)
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
    p=argparse.ArgumentParser("Time-global + Axial(z→y) Spatiotemporal Transformer: history -> future z-band @ step")
    # paths
    p.add_argument("--h5", type=str,
                   default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_interpolation_all_cases.h5")
    p.add_argument("--save-dir", type=str, default="./runs/timeGlobal_axialZY")

    # mode
    p.add_argument("--resume", default="")
    p.add_argument("--eval-only", action="store_true")

    # dataset / sampling
    p.add_argument("--tlength", type=int, default=81, help="输入时间窗口长度（覆盖 [step-tlength, step)）")
    p.add_argument("--zlength", type=int, default=5, help="历史/未来 z 带宽（长度均为 zlength）")
    p.add_argument("--time-stride", type=int, default=1, help="样本间 step 的步距")
    p.add_argument("--x-center-1based", type=int, default=21, help="中心 x 列（1-based），默认 21")
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

    # model dims
    p.add_argument("--d-model", type=int, default=256)

    # temporal encoder (global)
    p.add_argument("--t-nhead", type=int, default=8)
    p.add_argument("--t-layers", type=int, default=2)
    p.add_argument("--t-ffn-dim", type=int, default=1024)

    # axial spatial
    p.add_argument("--axial-nhead", type=int, default=4)
    p.add_argument("--axial-layers", type=int, default=1)
    p.add_argument("--axial-ffn-dim", type=int, default=512)

    # decoder
    p.add_argument("--dec-nhead", type=int, default=8)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--dec-ffn-dim", type=int, default=1024)

    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    # optional 2D refine
    p.add_argument("--use-2d-refine", action="store_true", help="启用 decoder 输出上的 2D Transformer 精炼")
    p.add_argument("--refine-layers", type=int, default=1)
    p.add_argument("--refine-nhead", type=int, default=4)
    p.add_argument("--refine-ffn-dim", type=int, default=512)
    p.add_argument("--refine-dropout", type=float, default=0.1)
    p.add_argument("--refine-act", type=str, default="gelu", choices=["relu","gelu"])

    # perf toggles
    p.add_argument("--amp", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true")

    # device & GPU
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu", type=int, default=-1)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="sec_hist2zfuture_timeGlobal_axial")
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

def _clone_split_with_stats(base: SectionBandCausalDataset, idx: List[int], normalize_flag: bool):
    ds = SectionBandCausalDataset(
        base.h5, tlength=base.tlength, zlength=base.zlength, time_stride=base.tstride,
        augment=False, normalize=normalize_flag, target_comp_idx=base.target_comp_idx,
        x_center_1based=(base.xc+1)
    )
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

    # datasets
    d_train_full = SectionBandCausalDataset(
        args.h5, tlength=args.tlength, zlength=args.zlength, time_stride=args.time_stride,
        augment=args.augment, normalize=(not args.no_normalize),
        target_comp_idx=target_idx, x_center_1based=args.x_center_1based
    )
    zlen, ny, nc = d_train_full.zlength, d_train_full._ny, len(target_idx)

    N=len(d_train_full); tr_idx, va_idx, te_idx = split_indices(N, args.train_ratio, args.val_ratio, args.seed)
    d_train=Subset(d_train_full, tr_idx)

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
    model = AxialSpatiotemporalTransformer(
        tlength=args.tlength, zlen=zlen, ny=ny, out_c=nc,
        d_model=args.d_model,
        t_nhead=args.t_nhead, t_layers=args.t_layers, t_ffn=args.t_ffn_dim,
        axial_nhead=args.axial_nhead, axial_layers=args.axial_layers, axial_ffn=args.axial_ffn_dim,
        dec_nhead=args.dec_nhead, dec_layers=args.dec_layers, dec_ffn=args.dec_ffn_dim,
        drop=args.dropout, act=args.act,
        use_spatial_refine=args.use_2d_refine,
        refine_layers=args.refine_layers, refine_nhead=args.refine_nhead,
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
