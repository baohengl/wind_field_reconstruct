#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate a trained Probes->Centerline Transformer on the validation split.

- Rebuilds the same dataset split recipe as training (same seed/ratios/time settings).
- Loads model from checkpoint (checkpoint_*.pt) or model_only.pt.
- Randomly samples N validation items and plots x/y/z components.
  Each figure is a 1x3 triptych: [GT, Reconstructed, Error].

Usage
-----
python validate_probes2centerline.py \
  --h5 /path/to/windfield_interpolation_all_cases.h5 \
  --ckpt /path/to/runs/probes2centerline/checkpoint_best.pt \
  --out-dir viz_val_center --num-samples 5 --seed 2025 --device cuda:0

If you only have model_only.pt (no args inside), specify arch via CLI and (optionally) nx/x_start:
  --from-model-only \
  --time-window 21 --d-model 160 --nhead 5 \
  --num-enc-layers 4 --num-dec-layers 2 --ffn-dim 640 --dropout 0.1 --act gelu \
  --nx-target 21 --x-start -1   # x-start -1 表示沿用中心窗口
"""

import os, math, random, argparse
from typing import List, Tuple, Optional, Dict
import h5py, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Attach a persistent h5 file handle to dataset inside (possibly nested) Subset."""
    if isinstance(ds, Subset):
        _attach_h5_recursive(ds.dataset)
    elif isinstance(ds, P2CDataset):
        if getattr(ds, "_h5", None) is None:
            ds._h5 = h5py.File(ds.h5, "r", swmr=True, libver="latest")

# ===================== Dataset (with normalization) =====================
class P2CDataset(Dataset):
    """
    Probes (time window) -> Centerline band at given z (single time t).
    Input per sample:
      probes_seq: (T_window, 9, 3)  -> flatten -> (T_window*9, 3)
    Target per sample:
      section slice: (Ny, nx_target, 3) with continuous columns [x_start : x_start + nx_target)

    归一化：
      - 从 /meta/section_velocity_mean (3,), /meta/section_velocity_std (3,) 读取
      - 对 probes 与 target 的最后一维 3 分量统一使用该统计量
    """
    def __init__(self, h5:str, time_window:int=21, z_start_1based:int=21,
                 nx_target:int=21, x_start:Optional[int]=None,
                 augment:bool=False, time_stride:int=1,
                 normalize:bool=True, eps:float=1e-6):
        assert time_window%2==1, "time_window must be odd"
        assert time_stride>=1, "time_stride must be >=1"
        assert nx_target>=1, "nx_target must be >=1"

        self.h5, self.tw, self.half = h5, time_window, time_window//2
        self.z0, self.augment, self.tstride = max(0,z_start_1based-1), augment, time_stride
        self.nx_target, self.x_start = nx_target, x_start
        self.items=[]  # (case,z,t)
        self._h5=None  # persistent handle

        # ---- 读取 meta 的 mean/std ----
        self.normalize = normalize
        self.eps = float(eps)
        self.norm_mean = np.zeros(3, dtype=np.float32)
        self.norm_std  = np.ones(3,  dtype=np.float32)
        try:
            with h5py.File(h5, "r") as fmeta:
                if normalize and "meta" in fmeta and \
                   "section_velocity_mean" in fmeta["meta"] and "section_velocity_std" in fmeta["meta"]:
                    m = fmeta["meta"]["section_velocity_mean"][...].astype(np.float32).reshape(3)
                    s = fmeta["meta"]["section_velocity_std"][...].astype(np.float32).reshape(3)
                    s = np.maximum(s, self.eps)
                    self.norm_mean[:] = m
                    self.norm_std[:]  = s
                else:
                    if normalize:
                        print("[Normalize] meta/{section_velocity_mean,std} 未找到，禁用归一化。")
                        self.normalize = False
        except Exception as e:
            if normalize:
                print(f"[Normalize] 读取 meta 失败（{e}），禁用归一化。")
                self.normalize = False

        with h5py.File(h5,"r") as f:
            if "cases" not in f: raise RuntimeError("HDF5 missing /cases")
            for cname,g in f["cases"].items():
                if not isinstance(g,h5py.Group) or ("section" not in g or "probes" not in g): continue
                Ts,Zs,Ny,Xs,Cs = g["section"].shape
                Tp,Zp,P9,Cp    = g["probes"].shape
                if not (Ts==Tp and Zs==Zp and Cp==3 and Cs==3 and P9==9):
                    raise RuntimeError(f"{cname}: shape mismatch - section {g['section'].shape}, probes {g['probes'].shape}")

                # compute X window（x_start<0 表示使用中心窗口）
                if self.x_start is None or (isinstance(self.x_start,int) and self.x_start<0):
                    center = Xs//2
                    half_w = (self.nx_target-1)//2
                    xs, xe = center - half_w, center + half_w + 1
                else:
                    xs = int(self.x_start)
                    xe = xs + self.nx_target

                if xs<0 or xe>Xs or (xe - xs)!=self.nx_target:
                    raise RuntimeError(f"{cname}: Invalid X slice [{xs}:{xe}) for Xs={Xs} and nx_target={self.nx_target}")

                self._xs, self._xe = xs, xe
                self._ny, self._nx_full = Ny, Xs

                # build indices
                z1, z2 = self.z0, Zs-1
                t1, t2 = self.half, Ts-1-self.half
                if z1>z2 or t1>t2: continue
                for z in range(z1, z2+1):
                    for t in range(t1, t2+1, self.tstride):
                        self.items.append((cname,z,t))

        if not self.items:
            raise RuntimeError("No valid (case,z,t); check time_window/z_start/time_stride.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i:int):
        cname,z,t=self.items[i]
        f = self._h5 if self._h5 is not None else h5py.File(self.h5,"r")
        g=f["cases"][cname]

        p=g["probes"][t-self.half:t+self.half+1, z, :, :].astype(np.float32)      # (T,9,3)
        s=g["section"][t, z, :, self._xs:self._xe, :].astype(np.float32)          # (Ny, nx_target, 3)

        # no augmentation for validation

        # 归一化（若启用）
        if self.normalize:
            p = (p - self.norm_mean) / self.norm_std
            s = (s - self.norm_mean) / self.norm_std

        p = np.ascontiguousarray(p); s = np.ascontiguousarray(s)
        p=torch.from_numpy(p.reshape(p.shape[0]*p.shape[1], 3))                   # (T*9,3)
        s=torch.from_numpy(s)                                                     # (Ny, nx_target, 3)
        meta={"case":cname,"z":z,"t":t}
        if self._h5 is None: f.close()
        return {"probes_seq":p, "target":s, "meta":meta}

# ===================== Model (aligned with training) =====================
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

class P2CTransformer(nn.Module):
    """
    Probes -> Centerline band (Ny, Nx_target, 3)
    (defaults match your training script)
    """
    def __init__(self,time_window:int, ny:int=49, nx_target:int=21,
                 d_model:int=160, nhead:int=5, enc_layers:int=4, dec_layers:int=2,
                 ffn:int=640, drop:float=0.1, act:str="gelu"):
        super().__init__()
        self.seq_len=time_window*9
        self.out_y,self.out_x,self.out_c=ny,nx_target,3
        nq=self.out_y*self.out_x

        self.in_proj=nn.Linear(3,d_model)
        self.pos=PosEnc(d_model, max_len=self.seq_len+8)

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

# ===================== Plotting =====================
def _triptych(axs, gt, pred, title_prefix:str, cmap_main:str="viridis", cmap_err:str="coolwarm"):
    """
    axs: [ax0, ax1, ax2] -> [GT, Pred, Err]
    gt, pred: 2D arrays (Ny x Nx_target)
    """
    err = pred - gt
    vmin = float(min(gt.min(), pred.min()))
    vmax = float(max(gt.max(), pred.max()))
    im0 = axs[0].imshow(gt, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_main)
    axs[0].set_title(f"{title_prefix} GT")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_main)
    axs[1].set_title(f"{title_prefix} Reconstructed")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    lim = float(max(abs(err.min()), abs(err.max())))
    if lim == 0.0: lim = 1e-8
    im2 = axs[2].imshow(err, origin="lower", aspect="auto", vmin=-lim, vmax=+lim, cmap=cmap_err)
    axs[2].set_title(f"{title_prefix} Error")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for a in axs:
        a.set_xlabel("x (Nx_target)"); a.set_ylabel("y (Ny)")

def plot_one_sample(out_dir:str, meta:dict, gt:np.ndarray, pred:np.ndarray):
    """
    gt, pred: (Ny, Nx_target, 3). Save three figures (Ux, Uy, Uz).
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{meta['case']}_z{meta['z']:03d}_t{meta['t']:05d}"
    comps = ["Ux","Uy","Uz"]
    for ci, cname in enumerate(comps):
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

def load_model_from_ckpt(ckpt_path:str,
                         device:str,
                         from_model_only:bool=False,
                         # fallbacks (only used when ckpt lacks args)
                         time_window:int=21, d_model:int=160, nhead:int=5,
                         num_enc_layers:int=4, num_dec_layers:int=2,
                         ffn_dim:int=640, dropout:float=0.1, act:str="gelu",
                         nx_target:int=21, ny:int=49, x_start:Optional[int]=None):
    """
    Returns:
      model(nn.Module), used(dict with arch & data slice), args_in(dict|None)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    args_in = None
    if (not from_model_only) and isinstance(ckpt, dict) and ("args" in ckpt):
        args_in = ckpt["args"]

    # arch优先从ckpt获取
    tw = _int_or_none((args_in or {}).get("time_window"), time_window)
    dm = _int_or_none((args_in or {}).get("d_model"), d_model)
    nh = _int_or_none((args_in or {}).get("nhead"), nhead)
    ne = _int_or_none((args_in or {}).get("num_enc_layers", (args_in or {}).get("num-enc-layers")), num_enc_layers)
    nd = _int_or_none((args_in or {}).get("num_dec_layers", (args_in or {}).get("num-dec-layers")), num_dec_layers)
    ff = _int_or_none((args_in or {}).get("ffn_dim"), ffn_dim)
    dr = float((args_in or {}).get("dropout", dropout))
    ac = str((args_in or {}).get("act", act))

    # 数据切片参数（与训练保持一致）
    nx_arg = (args_in or {}).get("nx_target", (args_in or {}).get("nx-target", nx_target))
    nx_use = _int_or_none(nx_arg, nx_target)
    xs_arg = (args_in or {}).get("x_start", x_start)
    xs_use = None if xs_arg is None else _int_or_none(xs_arg, x_start)

    # 构建模型
    model = P2CTransformer(tw, ny=ny, nx_target=nx_use,
                           d_model=dm, nhead=nh,
                           enc_layers=ne, dec_layers=nd,
                           ffn=ff, drop=dr, act=ac).to(device)

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

    used = dict(time_window=tw, d_model=dm, nhead=nh, num_enc_layers=ne, num_dec_layers=nd,
                ffn_dim=ff, dropout=dr, act=ac, nx_target=nx_use, x_start=xs_use, ny=ny)
    return model, used, args_in

def build_val_subset(h5:str, time_window:int, z_start_1based:int, time_stride:int,
                     train_ratio:float, val_ratio:float, seed:int,
                     nx_target:int, x_start:Optional[int],
                     normalize:bool=True):
    d_full = P2CDataset(h5, time_window, z_start_1based,
                        nx_target=nx_target, x_start=x_start,
                        augment=False, time_stride=time_stride,
                        normalize=normalize)
    N = len(d_full)
    tr_idx, va_idx, te_idx = split_indices(N, train_ratio, val_ratio, seed)
    d_val = Subset(d_full, va_idx)
    _attach_h5_recursive(d_val)  # keep h5 open for speed
    return d_val, d_full

# ===================== CLI & Main =====================
def parse_args():
    p=argparse.ArgumentParser("Validate probes->centerline model with visualization on validation split")
    # data / split (must match training to reproduce the same val set)
    p.add_argument("--h5", required=True, help="HDF5 path")
    p.add_argument("--time-window", type=int, default=21)
    p.add_argument("--z-start-1based", type=int, default=21)
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

    # override arch (only needed when --from-model-only)
    p.add_argument("--d-model", type=int, default=160)
    p.add_argument("--nhead", type=int, default=5)
    p.add_argument("--num-enc-layers", type=int, default=4)
    p.add_argument("--num-dec-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=640)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    # runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-samples", type=int, default=5, help="number of random val items to visualize")
    p.add_argument("--out-dir", type=str, default="viz_val_center")
    p.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on CUDA")

    # OPTIONAL: 手动覆盖（通常不用；ckpt 会带上）
    p.add_argument("--nx-target", type=int, default=None,
                   help="Manually override Nx_target if ckpt lacks it")
    p.add_argument("--x-start", type=int, default=None,
                   help="Override start column (0-based). If omitted/negative, use centered slice.")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed, strict_det=True)

    amp_on = (str(args.device).startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # 先用“保守默认值”构建一次 dataset，以便获知 Ny
    d_probe = P2CDataset(args.h5, args.time_window, args.z_start_1based,
                         nx_target=(args.nx_target if args.nx_target else 21),
                         x_start=args.x_start,
                         augment=False, time_stride=args.time_stride,
                         normalize=(not args.no_normalize))
    Ny = d_probe._ny

    # ---- 1) load model to get true arch + nx_target/x_start from ckpt ----
    model, used_arch, args_in = load_model_from_ckpt(
        ckpt_path=args.ckpt,
        device=args.device,
        from_model_only=args.from_model_only,
        time_window=args.time_window,
        d_model=args.d_model, nhead=args.nhead,
        num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers,
        ffn_dim=args.ffn_dim, dropout=args.dropout, act=args.act,
        nx_target=(args.nx_target if args.nx_target is not None else d_probe.nx_target),
        ny=Ny,
        x_start=(args.x_start if args.x_start is not None else d_probe.x_start if hasattr(d_probe, "x_start") else None)
    )
    model.eval()

    # 若 CLI 给了 time-window，但与 ckpt 不一致，则以 ckpt 为准并提示
    if used_arch["time_window"] != args.time_window:
        print(f"[Warn] Overriding --time-window {args.time_window} -> {used_arch['time_window']} to match checkpoint.")
    args.time_window = used_arch["time_window"]

    # 确定最终的 nx_target/x_start
    nx_target = used_arch["nx_target"] if args.nx_target is None else args.nx_target
    x_start   = used_arch["x_start"]   if args.x_start   is None else args.x_start

    print(f"[Arch] { {k:used_arch[k] for k in ['time_window','d_model','nhead','num_enc_layers','num_dec_layers','ffn_dim','dropout','act']} }")
    print(f"[DataSlice] Ny={Ny}, nx_target={nx_target}, x_start={x_start if x_start is not None else 'centered'}")

    # ---- 2) build validation subset with the aligned slice/window (with normalization) ----
    d_val, d_full = build_val_subset(
        h5=args.h5,
        time_window=args.time_window,
        z_start_1based=args.z_start_1based,
        time_stride=args.time_stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        nx_target=nx_target,
        x_start=x_start,
        normalize=(not args.no_normalize)
    )

    # 归一化统计量（用于反归一化作图）
    norm_mean = torch.from_numpy(d_full.norm_mean.astype(np.float32))
    norm_std  = torch.from_numpy(np.maximum(d_full.norm_std.astype(np.float32), 1e-6))

    # ---- 3) sample and visualize ----
    ns = min(args.num_samples, len(d_val))
    pick = sorted(random.sample(range(len(d_val)), ns))
    print(f"[Info] val size={len(d_val)}, sampled {ns} items -> {pick}")

    os.makedirs(args.out_dir, exist_ok=True)

    for i in tqdm(pick, desc="Visualizing"):
        item = d_val[i]
        x = item["probes_seq"].unsqueeze(0).to(args.device)  # (1, L, 3)  (已归一化)
        y = item["target"].unsqueeze(0).to(args.device)      # (1, Ny, nx_target, 3) (已归一化)

        # ensure seq_len match
        seq_len = x.shape[1]
        if hasattr(model, "seq_len") and model.seq_len != seq_len:
            print(f"[Warn] Overriding model.seq_len {model.seq_len} -> {seq_len} to match input.")
            model.seq_len = seq_len

        if amp_on:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                yhat = model(x)
        else:
            yhat = model(x)

        # 反归一化回物理单位作图： y_phys = y_norm * std + mean
        mean = norm_mean.to(args.device).view(1,1,1,3)
        std  = norm_std .to(args.device).view(1,1,1,3)

        gt_phys   = (y    * std + mean)[0].detach().cpu().numpy()
        pred_phys = (yhat * std + mean)[0].detach().cpu().numpy()

        plot_one_sample(args.out_dir, item["meta"], gt_phys, pred_phys)

    print(f"[Done] Figures saved to: {os.path.abspath(args.out_dir)}")

if __name__=="__main__":
    main()
