#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate a trained Probes->Center Z-band (center x column) Transformer on the validation split.

- Rebuilds the same dataset split recipe as training (same seed/ratios/time/z settings).
- Loads model from checkpoint (checkpoint_*.pt) or model_only.pt.
- Randomly samples N validation items and plots each requested component (e.g., y,z).
  Each figure is a 1x3 triptych: [GT, Reconstructed, Error], image shape (zlength x Ny).

Usage
-----
# 常规（不开启 2D 精炼）
python validate_probes2centerband.py \
  --h5 /path/to/data.h5 \
  --ckpt /path/to/runs/checkpoint_best.pt \
  --out-dir viz_val_centerband --num-samples 5 --seed 2025 --device cuda:0

# 仅有 model_only.pt（或 ckpt 不含 args），且需要开启 2D 精炼时需补齐结构参数
python validate_probes2centerband.py \
  --h5 /path/to/data.h5 \
  --ckpt /path/to/model_only.pt --from-model-only \
  --tlength 81 --zlength 9 \
  --d-model 256 --nhead 8 --num-enc-layers 6 --num-dec-layers 2 --ffn-dim 1024 --dropout 0.1 --act gelu \
  --target-comps yz --x-center-1based 21 \
  --use-2d-refine --refine-layers 2 --refine-nhead 4 --refine-ffn-dim 512 --refine-dropout 0.1 --refine-act gelu
"""

import os, math, random, argparse
from typing import List, Tuple, Optional, Dict
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== Utils =====================
def set_seed(seed:int, strict_det:bool=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = strict_det
    torch.backends.cudnn.benchmark = not strict_det

def split_indices(n:int, tr:float, vr:float, seed:int)->Tuple[List[int],List[int],List[int]]:
    idx=list(range(n)); rng=random.Random(seed); rng.shuffle(idx)
    ntr=int(n*tr); nvr=int(n*vr)
    return idx[:ntr], idx[ntr:ntr+nvr], idx[ntr+nvr:]

# ============ Persistent H5 helpers ============
def _attach_h5_recursive(ds):
    if isinstance(ds, Subset):
        _attach_h5_recursive(ds.dataset)
    elif isinstance(ds, P2CBandDataset):
        if getattr(ds, "_h5", None) is None:
            ds._h5 = h5py.File(ds.h5, "r", swmr=True, libver="latest")

# ===================== Dataset (aligned with training) =====================
class P2CBandDataset(Dataset):
    """
    Probes time-window ending at 'step' -> center-x z-band at time 'step'.

    Input  X: probes[step - tlength + 1 : step, z, 9, 3]  -> (tlength*9, 3)
    Target Y: section[step, z:z+zlength, :, x_center, comps] -> (zlength, Ny, C_out)
    """
    def __init__(self,
        h5:str,
        tlength:int=81,
        zlength:int=5,
        time_stride:int=1,
        augment:bool=False,            # not used in val
        normalize:bool=True,
        eps:float=1e-6,
        target_comp_idx: Optional[List[int]] = None, # default yz => [1,2]
        x_center_1based: Optional[int] = None,
    ):
        assert tlength>=1 and zlength>=1 and time_stride>=1
        self.h5 = h5
        self.tlength = int(tlength)
        self.zlength = int(zlength)
        self.tstride = int(time_stride)
        self.augment = bool(augment)
        self.items=[]; self._h5=None

        t_idx = [1,2] if not target_comp_idx else sorted(set(int(i) for i in target_comp_idx))
        if any(i not in (0,1,2) for i in t_idx): raise ValueError("target_comp_idx 仅允许 0/1/2")
        self.target_comp_idx = t_idx

        self.normalize = normalize
        self.eps = float(eps)
        self.norm_mean = np.zeros(3, dtype=np.float32)
        self.norm_std  = np.ones(3, dtype=np.float32)
        try:
            with h5py.File(h5, "r") as fmeta:
                if normalize and "meta" in fmeta and \
                   "section_velocity_mean" in fmeta["meta"] and "section_velocity_std" in fmeta["meta"]:
                    m = fmeta["meta"]["section_velocity_mean"][...].astype(np.float32).reshape(3)
                    s = np.maximum(fmeta["meta"]["section_velocity_std"][...].astype(np.float32).reshape(3), self.eps)
                    self.norm_mean[:] = m; self.norm_std[:] = s
                else:
                    if normalize: print("[Normalize] 未找到 meta/{section_velocity_mean,std}，禁用归一化。")
                    self.normalize = False
        except Exception as e:
            if normalize: print(f"[Normalize] 读取 meta 失败（{e}），禁用归一化。")
            self.normalize = False

        with h5py.File(h5,"r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 缺少 /cases")
            for cname,g in f["cases"].items():
                if not isinstance(g,h5py.Group) or ("section" not in g or "probes" not in g): continue
                Ts,Zs,Ny,Xs,Cs = g["section"].shape
                Tp,Zp,P9,Cp    = g["probes"].shape
                if not (Ts==Tp and Zs==Zp and Cs==3 and Cp==3 and P9==9):
                    raise RuntimeError(f"{cname}: shape mismatch - section {g['section'].shape}, probes {g['probes'].shape}")

                # center x column
                if x_center_1based is None:
                    xc = Xs // 2
                else:
                    xc = int(x_center_1based) - 1
                if not (0 <= xc < Xs):
                    raise RuntimeError(f"{cname}: x_center 索引无效 (xc={xc}, Xs={Xs})")
                self.xc = xc
                self._ny = Ny

                t_min = self.tlength - 1
                t_max = Ts - 1
                z_min = 0
                z_max = Zs - self.zlength
                if t_min > t_max or z_min > z_max: continue

                for z in range(z_min, z_max+1):
                    for step in range(t_min, t_max+1, self.tstride):
                        self.items.append((cname, z, step))

        if not self.items:
            raise RuntimeError("没有可用样本（检查 tlength/zlength/time_stride 是否过大）。")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname, z, step = self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5, "r")
        g = f["cases"][cname]

        t0 = step - self.tlength + 1
        t1 = step
        probes = g["probes"][t0:t1+1, z, :, :].astype(np.float32)                  # (tlength, 9, 3)
        sec    = g["section"][step, z:z+self.zlength, :, self.xc, :].astype(np.float32)  # (zlength, Ny, 3)
        target = sec[..., self.target_comp_idx]                                    # (zlength, Ny, C_out)

        if self.normalize:
            probes = (probes - self.norm_mean) / self.norm_std
            target = (target - self.norm_mean[self.target_comp_idx]) / np.maximum(self.norm_std[self.target_comp_idx], self.eps)

        x = torch.from_numpy(np.ascontiguousarray(probes.reshape(-1, 3)))          # (tlength*9, 3)
        y = torch.from_numpy(np.ascontiguousarray(target))                         # (zlength, Ny, C_out)
        meta={"case":cname,"z":z,"step":step}
        if self._h5 is None: f.close()
        return {"probes_seq":x, "target":y, "meta":meta}

# ===================== Model (aligned with training) =====================
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
    二维正弦位置编码：将 z 与 y 两个方向的 1D 正弦编码拼接为 D 维。
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
            pe[:, 1::2] = torch.cos(pos * div)[:, :-1]
        return pe  # (L, dim)

    def _build_2d_sincos(self, dz:int, dy:int, zlength:int, ny:int):
        pez = self._build_1d_sincos(zlength, dz)  # (Z, dz)
        pey = self._build_1d_sincos(ny, dy)       # (Y, dy)
        pe2d = torch.zeros(zlength, ny, dz + dy)
        for z in range(zlength):
            pe2d[z, :, :dz] = pez[z].unsqueeze(0).expand(ny, -1)
            pe2d[z, :, dz:] = pey
        return pe2d.reshape(zlength*ny, dz+dy)  # (Z*Y, D)

    def forward(self, tokens):  # tokens: (B, Z*Y, D)
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
    Encoder over probe time sequence; decoder queries (zlength*Ny) -> (zlength, Ny, C_out)
    可选：decoder 输出后添加 2D Transformer 精炼模块
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

        self.q = nn.Parameter(torch.randn(self.out_zy, d_model))
        self.qpos = nn.Parameter(torch.zeros(self.out_zy, d_model)); nn.init.normal_(self.qpos, std=0.02)

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
        h = self.in_proj(x).transpose(0,1)     # (L,B,D)
        h = self.encoder(self.pos(h))          # (L,B,D)
        q = (self.q + self.qpos).unsqueeze(1).expand(-1, B, -1)  # (out_zy, B, D)
        y = self.decoder(q, h).transpose(0,1)  # (B, out_zy, D)

        if self.srefine is not None:
            y = self.srefine(y)                # (B, out_zy, D)

        y = self.head(y)                       # (B, out_zy, C_out)
        return y.view(B, self.zlength, self.ny, self.out_c)

# ===================== Plotting =====================
def _triptych(axs, gt2d, pred2d, title_prefix:str, cmap_main:str="viridis", cmap_err:str="coolwarm"):
    """
    axs: [ax0, ax1, ax2] -> [GT, Pred, Err]
    gt2d, pred2d: arrays with shape (zlength, Ny). x-axis -> y, y-axis -> z-band index
    """
    err = pred2d - gt2d
    vmin = float(min(gt2d.min(), pred2d.min()))
    vmax = float(max(gt2d.max(), pred2d.max()))

    im0 = axs[0].imshow(gt2d, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_main)
    axs[0].set_title(f"{title_prefix} GT"); plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred2d, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_main)
    axs[1].set_title(f"{title_prefix} Reconstructed"); plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    lim = float(max(abs(err.min()), abs(err.max())))
    if lim == 0.0: lim = 1e-8
    im2 = axs[2].imshow(err, origin="lower", aspect="auto", vmin=-lim, vmax=+lim, cmap=cmap_err)
    axs[2].set_title(f"{title_prefix} Error"); plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for a in axs:
        a.set_xlabel("y index (Ny)")
        a.set_ylabel("z offset (0..zlength-1)")

def plot_one_sample(out_dir:str, meta:dict, gt:np.ndarray, pred:np.ndarray, comp_names:List[str]):
    """
    gt, pred: (zlength, Ny, C_out). Save one figure per component in comp_names.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{meta['case']}_z{meta['z']:03d}_t{meta['step']:05d}"
    for ci, cname in enumerate(comp_names):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        _triptych(axs, gt[:,:,ci], pred[:,:,ci], title_prefix=cname)
        fig.suptitle(f"{tag}", fontsize=10)
        fig.savefig(os.path.join(out_dir, f"{tag}_{cname}.png"), dpi=200)
        plt.close(fig)

# ===================== Loading & Split =====================
def _int_or_none(v, default=None):
    if v is None: return default
    try: return int(v)
    except Exception: return default

def _parse_target_comps(s: str)->List[int]:
    m={'x':0,'y':1,'z':2}
    s=(s or "yz").lower().strip()
    idx=sorted(set(m[c] for c in s if c in m))
    if not idx: raise ValueError(f"--target-comps '{s}' 无有效通道，示例：y / z / yz / xyz")
    return idx

def _comp_names_from_idx(idx: List[int])->List[str]:
    rev={0:"Ux",1:"Uy",2:"Uz"}
    return [rev[i] for i in idx]

def _get_bool_from_args_dict(d:Dict, *keys, default=False):
    for k in keys:
        if k in d: return bool(d[k])
    return default

def _get_val_from_args_dict(d:Dict, *keys, default=None, cast=None):
    for k in keys:
        if k in d:
            v = d[k]
            return cast(v) if (cast is not None) else v
    return default

def load_model_from_ckpt(ckpt_path:str,
                         device:str,
                         from_model_only:bool=False,
                         # fallbacks (only used when ckpt lacks args)
                         tlength:int=81, zlength:int=5,
                         d_model:int=256, nhead:int=8,
                         num_enc_layers:int=6, num_dec_layers:int=2,
                         ffn_dim:int=1024, dropout:float=0.1, act:str="gelu",
                         ny:int=49, target_comps:str="yz",
                         use_2d_refine:bool=False,
                         refine_layers:int=2, refine_nhead:int=4,
                         refine_ffn_dim:int=512, refine_dropout:float=0.1, refine_act:str="gelu"):
    """
    Returns: model(nn.Module), used(dict), args_in(dict|None)
    used contains: tlength, zlength, d_model, nhead, num_enc_layers, num_dec_layers, ffn_dim, dropout, act, ny, target_idx,
                   use_2d_refine, refine_layers, refine_nhead, refine_ffn_dim, refine_dropout, refine_act
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    args_in = None
    if (not from_model_only) and isinstance(ckpt, dict) and ("args" in ckpt):
        args_in = ckpt["args"]

    # arch优先从ckpt获取（兼容横线/下划线两种键名）
    tl = _int_or_none((args_in or {}).get("tlength"), tlength)
    zl = _int_or_none((args_in or {}).get("zlength"), zlength)
    dm = _int_or_none((args_in or {}).get("d_model"), d_model)
    nh = _int_or_none((args_in or {}).get("nhead"), nhead)
    ne = _int_or_none(_get_val_from_args_dict((args_in or {}), "num_enc_layers","num-enc-layers", default=num_enc_layers), num_enc_layers)
    nd = _int_or_none(_get_val_from_args_dict((args_in or {}), "num_dec_layers","num-dec-layers", default=num_dec_layers), num_dec_layers)
    ff = _int_or_none((args_in or {}).get("ffn_dim"), ffn_dim)
    dr = float((args_in or {}).get("dropout", dropout))
    ac = str((args_in or {}).get("act", act))
    tcomp = str(_get_val_from_args_dict((args_in or {}), "target_comps", "target-comps", default=target_comps))

    # refine（从 ckpt->args 优先，缺少则用 CLI 兜底）
    use_ref = _get_bool_from_args_dict((args_in or {}), "use_2d_refine","use-2d-refine", default=use_2d_refine)
    r_layers = _int_or_none(_get_val_from_args_dict((args_in or {}), "refine_layers","refine-layers", default=refine_layers), refine_layers)
    r_heads  = _int_or_none(_get_val_from_args_dict((args_in or {}), "refine_nhead","refine-nhead", default=refine_nhead), refine_nhead)
    r_ffn    = _int_or_none(_get_val_from_args_dict((args_in or {}), "refine_ffn_dim","refine-ffn-dim", default=refine_ffn_dim), refine_ffn_dim)
    r_drop   = float(_get_val_from_args_dict((args_in or {}), "refine_dropout","refine-dropout", default=refine_dropout))
    r_act    = str(_get_val_from_args_dict((args_in or {}), "refine_act","refine-act", default=refine_act))

    target_idx = _parse_target_comps(tcomp)
    nc = len(target_idx)

    model = P2CBandTransformer(
        tlength=tl, ny=ny, zlength=zl,
        d_model=dm, nhead=nh, enc_layers=ne, dec_layers=nd,
        ffn=ff, drop=dr, act=ac, out_c=nc,
        use_spatial_refine=use_ref,
        refine_nhead=r_heads, refine_layers=r_layers,
        refine_ffn=r_ffn, refine_drop=r_drop, refine_act=r_act
    ).to(device)

    # state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = {k:v for k,v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={missing}, unexpected={unexpected}")

    used = dict(tlength=tl, zlength=zl, d_model=dm, nhead=nh, num_enc_layers=ne, num_dec_layers=nd,
                ffn_dim=ff, dropout=dr, act=ac, ny=ny, target_idx=target_idx, target_comps=tcomp,
                use_2d_refine=use_ref, refine_layers=r_layers, refine_nhead=r_heads,
                refine_ffn_dim=r_ffn, refine_dropout=r_drop, refine_act=r_act)
    return model, used, args_in

def build_val_subset(h5:str,
                     tlength:int, zlength:int, time_stride:int,
                     train_ratio:float, val_ratio:float, seed:int,
                     target_idx: List[int],
                     x_center_1based: Optional[int],
                     normalize:bool=True):
    d_full = P2CBandDataset(
        h5, tlength=tlength, zlength=zlength, time_stride=time_stride,
        augment=False, normalize=normalize,
        target_comp_idx=target_idx, x_center_1based=x_center_1based
    )
    N = len(d_full)
    tr_idx, va_idx, te_idx = split_indices(N, train_ratio, val_ratio, seed)
    d_val = Subset(d_full, va_idx)
    _attach_h5_recursive(d_val)
    return d_val, d_full

# ===================== CLI & Main =====================
def parse_args():
    p=argparse.ArgumentParser("Validate probes->center Z-band model with visualization on validation split")
    # data / split
    p.add_argument("--h5", required=True, help="HDF5 path")
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2025)

    # normalization (align with training)
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable mean/std normalization even if meta stats exist.")

    # model ckpt
    p.add_argument("--ckpt", required=True, help="checkpoint_best.pt / checkpoint_last.pt / model_only.pt")
    p.add_argument("--from-model-only", action="store_true",
                   help="Set if --ckpt is model_only.pt or a bare state_dict w/o args")

    # override arch & dataset (only needed when --from-model-only or ckpt lacks args)
    p.add_argument("--tlength", type=int, default=81)
    p.add_argument("--zlength", type=int, default=5)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-enc-layers", type=int, default=6)
    p.add_argument("--num-dec-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    p.add_argument("--target-comps", type=str, default="yz",
                   help="目标分量（需与训练一致，如 yz / y / z / xyz）")
    p.add_argument("--x-center-1based", type=int, default=None,
                   help="中心 x 列（1-based），与训练一致；不设则默认取 X 的中间列")

    # 2D spatial refinement (必须与训练一致；ckpt 优先)
    p.add_argument("--use-2d-refine", action="store_true", help="启用 2D Transformer 空间精炼（decoder 后）")
    p.add_argument("--refine-layers", type=int, default=2, help="2D 精炼的 encoder 层数")
    p.add_argument("--refine-nhead", type=int, default=4, help="2D 精炼的注意力头数")
    p.add_argument("--refine-ffn-dim", type=int, default=512, help="2D 精炼的前馈维度")
    p.add_argument("--refine-dropout", type=float, default=0.1, help="2D 精炼的 dropout")
    p.add_argument("--refine-act", type=str, default="gelu", choices=["relu","gelu"], help="2D 精炼的激活函数")

    # runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-samples", type=int, default=5, help="number of random val items to visualize")
    p.add_argument("--out-dir", type=str, default="viz_val_centerband")
    p.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on CUDA")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed, strict_det=True)

    amp_on = (str(args.device).startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # 为探测 Ny & 提供 dataset 结构（如需）
    d_probe = P2CBandDataset(
        args.h5, tlength=args.tlength, zlength=args.zlength, time_stride=args.time_stride,
        augment=False, normalize=(not args.no_normalize),
        target_comp_idx=_parse_target_comps(args.target_comps),
        x_center_1based=args.x_center_1based
    )
    Ny = d_probe._ny

    # ---- 1) load model, prefer ckpt's args if available ----
    model, used_arch, args_in = load_model_from_ckpt(
        ckpt_path=args.ckpt,
        device=args.device,
        from_model_only=args.from_model_only,
        tlength=args.tlength, zlength=args.zlength,
        d_model=args.d_model, nhead=args.nhead,
        num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers,
        ffn_dim=args.ffn_dim, dropout=args.dropout, act=args.act,
        ny=Ny, target_comps=args.target_comps,
        use_2d_refine=args.use_2d_refine,
        refine_layers=args.refine_layers, refine_nhead=args.refine_nhead,
        refine_ffn_dim=args.refine_ffn_dim, refine_dropout=args.refine_dropout, refine_act=args.refine_act
    )
    model.eval()

    # 最终采用的参数
    tlength = used_arch["tlength"]
    zlength = used_arch["zlength"]
    target_idx = used_arch["target_idx"]
    comp_names = _comp_names_from_idx(target_idx)

    print(f"[Arch] {{'tlength': {tlength}, 'zlength': {zlength}, 'd_model': {used_arch['d_model']}, "
          f"'nhead': {used_arch['nhead']}, 'num_enc_layers': {used_arch['num_enc_layers']}, "
          f"'num_dec_layers': {used_arch['num_dec_layers']}, 'ffn_dim': {used_arch['ffn_dim']}, "
          f"'dropout': {used_arch['dropout']}, 'act': '{used_arch['act']}', 'ny': {used_arch['ny']}, "
          f"'target_comps': '{used_arch['target_comps']}', 'use_2d_refine': {used_arch['use_2d_refine']}, "
          f"'refine_layers': {used_arch['refine_layers']}, 'refine_nhead': {used_arch['refine_nhead']}, "
          f"'refine_ffn_dim': {used_arch['refine_ffn_dim']}, 'refine_dropout': {used_arch['refine_dropout']}, "
          f"'refine_act': '{used_arch['refine_act']}'}}")

    # ---- 2) build validation subset (aligned with training split) ----
    d_val, d_full = build_val_subset(
        h5=args.h5,
        tlength=tlength, zlength=zlength, time_stride=args.time_stride,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
        target_idx=target_idx,
        x_center_1based=args.x_center_1based,
        normalize=(not args.no_normalize)
    )

    # 归一化统计量（用于反归一化作图）
    mean_all = torch.from_numpy(d_full.norm_mean.astype(np.float32))  # (3,)
    std_all  = torch.from_numpy(np.maximum(d_full.norm_std.astype(np.float32), 1e-6))  # (3,)
    mean_sel = mean_all[target_idx].view(1,1,1,len(target_idx))
    std_sel  = std_all [target_idx].view(1,1,1,len(target_idx))

    # ---- 3) sample and visualize ----
    ns = min(args.num_samples, len(d_val))
    pick = sorted(random.sample(range(len(d_val)), ns))
    print(f"[Info] val size={len(d_val)}, sampled {ns} items -> {pick}")

    os.makedirs(args.out_dir, exist_ok=True)

    for i in tqdm(pick, desc="Visualizing"):
        item = d_val[i]
        x = item["probes_seq"].unsqueeze(0).to(args.device)   # (1, L, 3)
        y = item["target"].unsqueeze(0).to(args.device)       # (1, zlength, Ny, C)

        # ensure seq_len compatibility (for safety)
        seq_len = x.shape[1]
        if hasattr(model, "seq_len") and model.seq_len != seq_len:
            print(f"[Warn] Overriding model.seq_len {model.seq_len} -> {seq_len} to match input.")
            model.seq_len = seq_len

        if amp_on:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                yhat = model(x)  # (1, zlength, Ny, C)
        else:
            yhat = model(x)

        # 反归一化
        gt_phys   = (y    * std_sel.to(y.device) + mean_sel.to(y.device))[0].detach().cpu().numpy()
        pred_phys = (yhat * std_sel.to(y.device) + mean_sel.to(y.device))[0].detach().cpu().numpy()

        plot_one_sample(args.out_dir, item["meta"], gt_phys, pred_phys, comp_names)

    print(f"[Done] Figures saved to: {os.path.abspath(args.out_dir)}")

if __name__=="__main__":
    main()
