#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate "Time-Global + Spatial Axial (z→y)" Transformer on validation split,
with GT/Pred/Error triptychs for each selected component (e.g., y,z).

特点
-----
- 与训练脚本完全对齐的数据切片与 split（相同 seed / ratios / tlength / zlength / x_center）。
- 优先从 ckpt['args'] 读取结构与数据参数；若是 model_only.pt，则通过 CLI 指定。
- 随机抽取 N 个验证样本，对每个通道输出 1x3 图：[GT, Pred, Error]，尺寸 (zlength x Ny)。

用法示例
--------
# 直接加载带 args 的 ckpt
python validate_timeGlobal_axialZY.py \
  --h5 /vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_interpolation_all_cases.h5 \
  --ckpt ./runs/timeGlobal_axialZY_v1/checkpoint_best.pt \
  --out-dir viz_val_axial --num-samples 6 --seed 2025 --device cuda:0

# 仅有 model_only.pt，需要补齐结构参数
python validate_timeGlobal_axialZY.py \
  --h5 /.../windfield_interpolation_all_cases.h5 \
  --ckpt ./runs/timeGlobal_axialZY_v1/model_only.pt --from-model-only \
  --tlength 81 --zlength 5 \
  --d-model 256 \
  --t-nhead 8 --t-layers 2 --t-ffn-dim 1024 \
  --axial-nhead 4 --axial-layers 1 --axial-ffn-dim 512 \
  --dec-nhead 8 --dec-layers 2 --dec-ffn-dim 1024 \
  --dropout 0.1 --act gelu \
  --target-comps yz --x-center-1based 21 \
  --use-2d-refine --refine-layers 1 --refine-nhead 4 --refine-ffn-dim 512
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

# ===================== Dataset (aligned with training) =====================
class SectionBandCausalDataset(Dataset):
    """
    与训练脚本一致的切片：
      step ∈ [Tlength, T-1]
      z    ∈ [zlength, Z - zlength - 1]

    X = section[step-Tlength:step, z-zlength:z, :, x_center, comps]   -> (Tlength, zlength, Ny, C)
    Y = section[step,           z+1:z+zlength, :, x_center, comps]    -> (zlength, Ny, C)
    """
    def __init__(self,
        h5:str,
        tlength:int=81,
        zlength:int=5,
        time_stride:int=1,
        augment:bool=False,            # 验证不使用
        normalize:bool=True,
        eps:float=1e-6,
        target_comp_idx: Optional[List[int]] = None,  # 默认 yz -> [1,2]
        x_center_1based: Optional[int] = 21,
    ):
        assert tlength>=1 and zlength>=1 and time_stride>=1
        self.h5 = h5
        self.tlength = int(tlength)
        self.zlength = int(zlength)
        self.tstride = int(time_stride)
        self.augment = bool(augment)
        self.items=[]; self._h5=None

        t_idx = [1,2] if not target_comp_idx else sorted(set(int(i) for i in target_comp_idx))
        if any(i not in (0,1,2) for i in t_idx):
            raise ValueError("target_comp_idx 仅允许 0/1/2（对应 x/y/z）")
        self.target_comp_idx = t_idx

        # norm stats
        self.normalize = normalize
        self.eps = float(eps)
        self.norm_mean = np.zeros(3, dtype=np.float32)
        self.norm_std  = np.ones(3, dtype=np.float32)
        try:
            with h5py.File(h5, "r") as fm:
                if normalize and "meta" in fm and \
                   "section_velocity_mean" in fm["meta"] and "section_velocity_std" in fm["meta"]:
                    m = fm["meta"]["section_velocity_mean"][...].astype(np.float32).reshape(3)
                    s = np.maximum(fm["meta"]["section_velocity_std"][...].astype(np.float32).reshape(3), self.eps)
                    self.norm_mean[:] = m; self.norm_std[:] = s
                else:
                    if normalize: print("[Normalize] 未找到 meta/{section_velocity_mean,std}，禁用归一化。")
                    self.normalize = False
        except Exception as e:
            if normalize: print(f"[Normalize] 读取 meta 失败（{e}），禁用归一化。")
            self.normalize = False

        with h5py.File(h5, "r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 缺少 /cases")
            for cname, g in f["cases"].items():
                if not isinstance(g, h5py.Group) or ("section" not in g):
                    continue
                T, Z, Ny, Nx, C = g["section"].shape
                if C != 3:
                    raise RuntimeError(f"{cname}: section[...,3] 预期为 3 分量，得到 {C}")

                xc = (Nx // 2) if x_center_1based is None else int(x_center_1based) - 1
                if not (0 <= xc < Nx):
                    raise RuntimeError(f"{cname}: x_center 无效 (xc={xc}, Nx={Nx})")
                self.xc = xc
                self._ny = Ny

                step_min = self.tlength
                step_max = T - 1
                z_min = self.zlength
                z_max = Z - self.zlength - 1
                if step_min > step_max or z_min > z_max: continue

                for z in range(z_min, z_max+1):
                    for step in range(step_min, step_max+1, self.tstride):
                        self.items.append((cname, z, step))

        if not self.items:
            raise RuntimeError("没有可用样本（检查 tlength/zlength/time_stride 是否过大）。")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname, z, step = self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5, "r")
        g = f["cases"][cname]

        t0 = step - self.tlength
        t1 = step
        zh0 = z - self.zlength
        zh1 = z
        zf0 = z + 1
        zf1 = z + 1 + self.zlength

        x = g["section"][t0:t1, zh0:zh1, :, self.xc, :].astype(np.float32)  # (T,Zhist,Ny,3)
        x = x[..., self.target_comp_idx]                                     # (T,Zhist,Ny,C)
        y = g["section"][step, zf0:zf1, :, self.xc, :].astype(np.float32)    # (Zout,Ny,3)
        y = y[..., self.target_comp_idx]                                     # (Zout,Ny,C)

        if self.normalize:
            mean = self.norm_mean[self.target_comp_idx]
            std  = np.maximum(self.norm_std[self.target_comp_idx], self.eps)
            x = (x - mean) / std
            y = (y - mean) / std

        X = torch.from_numpy(np.ascontiguousarray(x))  # (T, Zhist, Ny, C)
        Y = torch.from_numpy(np.ascontiguousarray(y))  # (Zout, Ny, C)
        meta={"case":cname,"z":z,"step":step}
        if self._h5 is None: f.close()
        return {"hist": X, "target": Y, "meta": meta}

# ===================== Model (aligned with training) =====================
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
        pez = self.pez.to(x.device).to(x.dtype)
        pey = self.pey.to(x.device).to(x.dtype)
        pe = torch.zeros(zlen, ny, D, device=x.device, dtype=x.dtype)
        pe[:,:, :dz] = pez.unsqueeze(1).expand(-1, ny, -1)
        pe[:,:, dz:] = pey.unsqueeze(0).expand(zlen, -1, -1)
        pe = pe.reshape(1, zlen*ny, D)
        return x + pe

def causal_mask(T:int, device):
    m = torch.full((T, T), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m

class TemporalTransformerEncoder(nn.Module):
    """Time-global Transformer over length-T sequences at each (z,y)."""
    def __init__(self, c_in:int, d_model:int, nhead:int=8, layers:int=2, ffn:int=1024, drop:float=0.1, act:str="gelu"):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
    def forward(self, x):  # x: (B*Z*Y, T, C_in)
        BZY, T, _ = x.shape
        h = self.proj(x)
        mask = causal_mask(T, x.device)
        h = self.encoder(h, mask=mask)
        return h[:, -1, :]  # last step feature

class AxialSelfAttentionZY(nn.Module):
    """轴向自注意力：先沿 z（每条 y 线），再沿 y（每条 z 线）"""
    def __init__(self, d_model:int, nhead:int=4, layers:int=1, ffn:int=512, drop:float=0.1, act:str="gelu"):
        super().__init__()
        enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=True, norm_first=True)
        self.z_block = nn.TransformerEncoder(enc_ly, layers)
        self.y_block = nn.TransformerEncoder(enc_ly, layers)
    def forward(self, mem):  # (B,Z,Y,D)
        B,Z,Y,D = mem.shape
        h = mem.permute(0,2,1,3).reshape(B*Y, Z, D)  # z-attn per y
        h = self.z_block(h)
        h = h.reshape(B, Y, Z, D).permute(0,2,1,3)   # (B,Z,Y,D)
        h2 = h.reshape(B*Z, Y, D)                    # y-attn per z
        h2 = self.y_block(h2)
        return h2.reshape(B, Z, Y, D)

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
                 d_model:int=256,
                 t_nhead:int=8, t_layers:int=2, t_ffn:int=1024,
                 axial_nhead:int=4, axial_layers:int=1, axial_ffn:int=512,
                 dec_nhead:int=8, dec_layers:int=2, dec_ffn:int=1024,
                 drop:float=0.1, act:str="gelu", use_spatial_refine:bool=False,
                 refine_layers:int=1, refine_nhead:int=4, refine_ffn:int=512, refine_drop:float=0.1, refine_act:str="gelu"):
        super().__init__()
        self.tlength, self.zlen, self.ny, self.out_c = tlength, zlen, ny, out_c
        self.d_model = d_model

        # time-global encoder
        self.temp_enc = TemporalTransformerEncoder(out_c, d_model, t_nhead, t_layers, t_ffn, drop, act)
        # spatial axial z→y
        self.axial = AxialSelfAttentionZY(d_model, axial_nhead, axial_layers, axial_ffn, drop, act)

        # decoder (queries cross-attend to memory)
        dec_ly = nn.TransformerDecoderLayer(d_model, dec_nhead, dec_ffn, drop, activation=act,
                                            batch_first=False, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_ly, dec_layers)

        # positions & queries
        self.mem_pos = PosEnc2D(d_model, zlen, ny)
        self.q = nn.Parameter(torch.randn(zlen*ny, d_model))
        nn.init.normal_(self.q, std=0.02)
        self.q_pos = PosEnc2D(d_model, zlen, ny)

        # optional refine
        self.use_spatial_refine = bool(use_spatial_refine)
        if self.use_spatial_refine:
            self.srefine = SpatialRefiner2D(d_model, zlen, ny,
                                            nhead=refine_nhead, num_layers=refine_layers,
                                            ffn=refine_ffn, drop=refine_drop, act=refine_act)
        else:
            self.srefine = None

        self.head = nn.Linear(d_model, out_c)

    def forward(self, x):  # (B, T, Zhist, Ny, C)
        B,T,Z,Y,C = x.shape
        h = x.permute(0,2,3,1,4).reshape(B*Z*Y, T, C)
        h = self.temp_enc(h).view(B, Z, Y, self.d_model)
        h = self.axial(h)                                     # (B,Z,Y,D)

        mem = h.reshape(B, Z*Y, self.d_model)
        mem = self.mem_pos(mem).transpose(0,1)                # (Z*Y,B,D)

        q = self.q.unsqueeze(0).expand(B, -1, -1)
        q = self.q_pos(q).transpose(0,1)                      # (Z*Y,B,D)

        y = self.decoder(q, mem).transpose(0,1)               # (B,Z*Y,D)
        if self.srefine is not None:
            y = self.srefine(y)
        y = self.head(y).view(B, self.zlen, self.ny, self.out_c)
        return y

# ===================== Plotting =====================
def _triptych(axs, gt2d, pred2d, title_prefix:str, cmap_main:str="viridis", cmap_err:str="coolwarm"):
    err = pred2d - gt2d
    vmin = float(min(gt2d.min(), pred2d.min()))
    vmax = float(max(gt2d.max(), pred2d.max()))
    im0 = axs[0].imshow(gt2d, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_main)
    axs[0].set_title(f"{title_prefix} GT"); plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(pred2d, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_main)
    axs[1].set_title(f"{title_prefix} Pred"); plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    lim = float(max(abs(err.min()), abs(err.max()))); lim = lim if lim>0 else 1e-8
    im2 = axs[2].imshow(err, origin="lower", aspect="auto", vmin=-lim, vmax=+lim, cmap=cmap_err)
    axs[2].set_title(f"{title_prefix} Error"); plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    for a in axs:
        a.set_xlabel("y index (Ny)")
        a.set_ylabel("z offset (0..zlength-1)")

def plot_one_sample(out_dir:str, meta:dict, gt:np.ndarray, pred:np.ndarray, comp_names:List[str]):
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

def _parse_target_comps(s: str)->List[int]:
    m={'x':0,'y':1,'z':2}
    s=(s or "yz").lower().strip()
    idx=sorted(set(m[c] for c in s if c in m))
    if not idx: raise ValueError(f"--target-comps '{s}' 无有效通道，示例：y / z / yz / xyz")
    return idx

def _comp_names_from_idx(idx: List[int])->List[str]:
    rev={0:"Ux",1:"Uy",2:"Uz"}
    return [rev[i] for i in idx]

def load_model_from_ckpt(ckpt_path:str,
                         device:str,
                         from_model_only:bool=False,
                         # fallbacks (only used when ckpt lacks args)
                         tlength:int=81, zlength:int=5,
                         d_model:int=256,
                         t_nhead:int=8, t_layers:int=2, t_ffn_dim:int=1024,
                         axial_nhead:int=4, axial_layers:int=1, axial_ffn_dim:int=512,
                         dec_nhead:int=8, dec_layers:int=2, dec_ffn_dim:int=1024,
                         dropout:float=0.1, act:str="gelu",
                         ny:int=49, target_comps:str="yz",
                         use_2d_refine:bool=False,
                         refine_layers:int=1, refine_nhead:int=4,
                         refine_ffn_dim:int=512, refine_dropout:float=0.1, refine_act:str="gelu"):
    """
    返回: model(nn.Module), used(dict), args_in(dict|None)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_in = None
    if (not from_model_only) and isinstance(ckpt, dict) and ("args" in ckpt):
        args_in = ckpt["args"]

    # 从 ckpt -> args 读取（没有则用 CLI fallback）
    tl = _int_or_none((args_in or {}).get("tlength"), tlength)
    zl = _int_or_none((args_in or {}).get("zlength"), zlength)
    dm = _int_or_none((args_in or {}).get("d_model"), d_model)

    tn = _int_or_none(_get_val_from_args_dict((args_in or {}),"t_nhead","t-nhead", default=t_nhead), t_nhead)
    tlr= _int_or_none(_get_val_from_args_dict((args_in or {}),"t_layers","t-layers", default=t_layers), t_layers)
    tff= _int_or_none(_get_val_from_args_dict((args_in or {}),"t_ffn_dim","t-ffn-dim", default=t_ffn_dim), t_ffn_dim)

    an = _int_or_none(_get_val_from_args_dict((args_in or {}),"axial_nhead","axial-nhead", default=axial_nhead), axial_nhead)
    alr= _int_or_none(_get_val_from_args_dict((args_in or {}),"axial_layers","axial-layers", default=axial_layers), axial_layers)
    aff= _int_or_none(_get_val_from_args_dict((args_in or {}),"axial_ffn_dim","axial-ffn-dim", default=axial_ffn_dim), axial_ffn_dim)

    dn = _int_or_none(_get_val_from_args_dict((args_in or {}),"dec_nhead","dec-nhead", default=dec_nhead), dec_nhead)
    dlr= _int_or_none(_get_val_from_args_dict((args_in or {}),"dec_layers","dec-layers", default=dec_layers), dec_layers)
    dff= _int_or_none(_get_val_from_args_dict((args_in or {}),"dec_ffn_dim","dec-ffn-dim", default=dec_ffn_dim), dec_ffn_dim)

    dr = float((args_in or {}).get("dropout", dropout))
    ac = str((args_in or {}).get("act", act))
    tcomp = str(_get_val_from_args_dict((args_in or {}), "target_comps","target-comps", default=target_comps))

    # refine
    use_ref = _get_bool_from_args_dict((args_in or {}), "use_2d_refine","use-2d-refine", default=use_2d_refine)
    r_layers = _int_or_none(_get_val_from_args_dict((args_in or {}), "refine_layers","refine-layers", default=refine_layers), refine_layers)
    r_heads  = _int_or_none(_get_val_from_args_dict((args_in or {}), "refine_nhead","refine-nhead", default=refine_nhead), refine_nhead)
    r_ffn    = _int_or_none(_get_val_from_args_dict((args_in or {}), "refine_ffn_dim","refine-ffn-dim", default=refine_ffn_dim), refine_ffn_dim)
    r_drop   = float(_get_val_from_args_dict((args_in or {}), "refine_dropout","refine-dropout", default=refine_dropout))
    r_act    = str(_get_val_from_args_dict((args_in or {}), "refine_act","refine-act", default=refine_act))

    target_idx = _parse_target_comps(tcomp)
    nc = len(target_idx)

    model = AxialSpatiotemporalTransformer(
        tlength=tl, zlen=zl, ny=ny, out_c=nc,
        d_model=dm,
        t_nhead=tn, t_layers=tlr, t_ffn=tff,
        axial_nhead=an, axial_layers=alr, axial_ffn=aff,
        dec_nhead=dn, dec_layers=dlr, dec_ffn=dff,
        drop=dr, act=ac,
        use_spatial_refine=use_ref,
        refine_layers=r_layers, refine_nhead=r_heads,
        refine_ffn=r_ffn, refine_drop=r_drop, refine_act=r_act
    ).to(device)

    # 加载 state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = {k:v for k,v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={missing}, unexpected={unexpected}")

    used = dict(
        tlength=tl, zlength=zl, d_model=dm,
        t_nhead=tn, t_layers=tlr, t_ffn_dim=tff,
        axial_nhead=an, axial_layers=alr, axial_ffn_dim=aff,
        dec_nhead=dn, dec_layers=dlr, dec_ffn_dim=dff,
        dropout=dr, act=ac, ny=ny, target_idx=target_idx, target_comps=tcomp,
        use_2d_refine=use_ref, refine_layers=r_layers, refine_nhead=r_heads,
        refine_ffn_dim=r_ffn, refine_dropout=r_drop, refine_act=r_act
    )
    return model, used, args_in

def build_val_subset(h5:str,
                     tlength:int, zlength:int, time_stride:int,
                     train_ratio:float, val_ratio:float, seed:int,
                     target_idx: List[int],
                     x_center_1based: Optional[int],
                     normalize:bool=True):
    d_full = SectionBandCausalDataset(
        h5, tlength=tlength, zlength=zlength, time_stride=time_stride,
        augment=False, normalize=normalize,
        target_comp_idx=target_idx, x_center_1based=x_center_1based
    )
    N = len(d_full)
    tr_idx, va_idx, te_idx = split_indices(N, train_ratio, val_ratio, seed)
    d_val = Subset(d_full, va_idx)
    return d_val, d_full

# ===================== CLI & Main =====================
def parse_args():
    p=argparse.ArgumentParser("Validate Time-Global + Axial(z→y) model with visualization on validation split")
    # data / split
    p.add_argument("--h5", required=True, help="HDF5 path")
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2025)

    # normalization (align with training)
    p.add_argument("--no-normalize", action="store_true")

    # model ckpt
    p.add_argument("--ckpt", required=True, help="checkpoint_best.pt / checkpoint_last.pt / model_only.pt")
    p.add_argument("--from-model-only", action="store_true",
                   help="若为 model_only.pt 或 ckpt 不含 args，请显式提供结构参数")

    # overrides when needed (only used if ckpt lacks args)
    p.add_argument("--tlength", type=int, default=81)
    p.add_argument("--zlength", type=int, default=5)
    p.add_argument("--d-model", type=int, default=256)

    p.add_argument("--t-nhead", type=int, default=8)
    p.add_argument("--t-layers", type=int, default=2)
    p.add_argument("--t-ffn-dim", type=int, default=1024)

    p.add_argument("--axial-nhead", type=int, default=4)
    p.add_argument("--axial-layers", type=int, default=1)
    p.add_argument("--axial-ffn-dim", type=int, default=512)

    p.add_argument("--dec-nhead", type=int, default=8)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--dec-ffn-dim", type=int, default=1024)

    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    p.add_argument("--target-comps", type=str, default="yz")
    p.add_argument("--x-center-1based", type=int, default=21)

    # 2D spatial refinement (must match training)
    p.add_argument("--use-2d-refine", action="store_true")
    p.add_argument("--refine-layers", type=int, default=1)
    p.add_argument("--refine-nhead", type=int, default=4)
    p.add_argument("--refine-ffn-dim", type=int, default=512)
    p.add_argument("--refine-dropout", type=float, default=0.1)
    p.add_argument("--refine-act", type=str, default="gelu", choices=["relu","gelu"])

    # runtime & io
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--out-dir", type=str, default="viz_val_axial")
    p.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on CUDA")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed, strict_det=True)

    # 先构建一个 dataset 来探测 Ny（必要）
    d_probe = SectionBandCausalDataset(
        args.h5, tlength=args.tlength, zlength=args.zlength, time_stride=args.time_stride,
        augment=False, normalize=(not args.no_normalize),
        target_comp_idx=_parse_target_comps(args.target_comps),
        x_center_1based=args.x_center_1based
    )
    Ny = d_probe._ny

    # ---- 1) load model (ckpt args 优先) ----
    model, used, args_in = load_model_from_ckpt(
        ckpt_path=args.ckpt,
        device=args.device,
        from_model_only=args.from_model_only,
        tlength=args.tlength, zlength=args.zlength,
        d_model=args.d_model,
        t_nhead=args.t_nhead, t_layers=args.t_layers, t_ffn_dim=args.t_ffen_dim if hasattr(args,'t_ffen_dim') else args.t_ffn_dim,
        axial_nhead=args.axial_nhead, axial_layers=args.axial_layers, axial_ffn_dim=args.axial_ffn_dim,
        dec_nhead=args.dec_nhead, dec_layers=args.dec_layers, dec_ffn_dim=args.dec_ffn_dim,
        dropout=args.dropout, act=args.act,
        ny=Ny, target_comps=args.target_comps,
        use_2d_refine=args.use_2d_refine,
        refine_layers=args.refine_layers, refine_nhead=args.refine_nhead,
        refine_ffn_dim=args.refine_ffn_dim, refine_dropout=args.refine_dropout, refine_act=args.refine_act
    )
    model.eval()

    # 最终采用的关键参数
    tlength = used["tlength"]; zlength = used["zlength"]
    target_idx = used["target_idx"]; comp_names = _comp_names_from_idx(target_idx)

    print("[Arch used]", {k: used[k] for k in [
        "tlength","zlength","d_model",
        "t_nhead","t_layers","t_ffn_dim",
        "axial_nhead","axial_layers","axial_ffn_dim",
        "dec_nhead","dec_layers","dec_ffn_dim",
        "dropout","act","ny","target_comps",
        "use_2d_refine","refine_layers","refine_nhead","refine_ffn_dim","refine_dropout","refine_act"
    ]})

    # ---- 2) build validation subset (aligned with training split) ----
    d_val, d_full = build_val_subset(
        h5=args.h5,
        tlength=tlength, zlength=zlength, time_stride=args.time_stride,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
        target_idx=target_idx,
        x_center_1based=args.x_center_1based,
        normalize=(not args.no_normalize)
    )

    # 反归一化统计
    # ✅ 正确的写法：从 d_val 拿到底层 Dataset 的统计量
    if isinstance(d_val, Subset):
        base_ds = d_val.dataset  # SectionBandCausalDataset
    else:
        base_ds = d_val          # 本身就是 Dataset
    mean_all = torch.from_numpy(base_ds.norm_mean.astype(np.float32))
    std_all  = torch.from_numpy(np.maximum(base_ds.norm_std.astype(np.float32), 1e-6))
    mean_sel = mean_all[target_idx].view(1,1,1,len(target_idx))
    std_sel  = std_all [target_idx].view(1,1,1,len(target_idx))

    # ---- 3) sample & visualize ----
    ns = min(args.num_samples, len(d_val))
    pick = sorted(random.sample(range(len(d_val)), ns))
    print(f"[Info] val size={len(d_val)}, sampled {ns} items -> {pick}")

    os.makedirs(args.out_dir, exist_ok=True)
    amp_on = str(args.device).startswith("cuda")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    for i in tqdm(pick, desc="Visualizing"):
        item = d_val[i]
        X = item["hist"].unsqueeze(0).to(args.device)     # (1,T,Z,Ny,C)
        Y = item["target"].unsqueeze(0).to(args.device)   # (1,Z,Ny,C)

        if amp_on:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                Yhat = model(X)
        else:
            Yhat = model(X)

        # 反归一化到物理量级
        gt_phys   = (Y    * std_sel.to(Y.device) + mean_sel.to(Y.device))[0].detach().cpu().numpy()
        pred_phys = (Yhat * std_sel.to(Y.device) + mean_sel.to(Y.device))[0].detach().cpu().numpy()

        plot_one_sample(args.out_dir, item["meta"], gt_phys, pred_phys, comp_names)

    print(f"[Done] Figures saved to: {os.path.abspath(args.out_dir)}")

if __name__=="__main__":
    main()
