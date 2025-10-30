#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatiotemporal Transformer (Three-Axis 1D Attention: T, Z, Y)
Encoder-only 3D PE with pre-registered flat buffer (T*Z*Y,D).
Decoder memory: NO positional encoding (q keeps 2D qpos).
Optional 2D refine preserved.

Input  X: (Tlength, zlength, Ny, C)
Target Y: (zlength, Ny, C)
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
    X = section[step-Tlength:step, z-zlength:z, :, x_center, comps] -> (Tlength, zlength, Ny, C)
    Y = section[step,           z+1:z+zlength, :, x_center, comps]  -> (zlength, Ny, C)
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

# ===================== Positional Encodings =====================
class PosEnc1D:
    @staticmethod
    def build(d: int, L: int) -> torch.Tensor:
        pe = torch.zeros(L, d)
        pos = torch.arange(L).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div[:pe[:,1::2].shape[1]])
        return pe  # (L,d), float32

class FlatPosEnc3D(nn.Module):
    """
    3D 正弦位置编码的展平版本，注册为 buffer: pe_flat (T*Z*Y, D)
    用法：输入 (B,T,Z,Y,D) -> 展平为 (B,T*Z*Y,D) 后做 x + pe_flat[None] -> 再 reshape 回来
    """
    def __init__(self, d_model:int, tlen:int, zlen:int, ny:int):
        super().__init__()
        dt = d_model // 3
        dz = d_model // 3
        dy = d_model - dt - dz
        pet = PosEnc1D.build(dt, tlen)  # (T,dt)
        pez = PosEnc1D.build(dz, zlen) # (Z,dz)
        pey = PosEnc1D.build(dy, ny)   # (Y,dy)

        # 构建一次 (T,Z,Y,D) 的 PE，并登记其展平形态为 buffer
        T, Z, Y = tlen, zlen, ny
        pe = torch.zeros(T, Z, Y, d_model, dtype=torch.float32)
        pe[:,:,:, :dt]       = pet.unsqueeze(1).unsqueeze(2).expand(T, Z, Y, dt)
        pe[:,:,:, dt:dt+dz]  = pez.unsqueeze(0).unsqueeze(2).expand(T, Z, Y, dz)
        pe[:,:,:, dt+dz: ]   = pey.unsqueeze(0).unsqueeze(1).expand(T, Z, Y, dy)
        pe_flat = pe.view(T*Z*Y, d_model)  # (T*Z*Y, D)
        self.register_buffer("pe_flat", pe_flat, persistent=True)
        self.tlen, self.zlen, self.ny, self.d_model = tlen, zlen, ny, d_model

    def add_pe(self, x_btzyd: torch.Tensor) -> torch.Tensor:
        """
        x_btzyd: (B,T,Z,Y,D)  (任意 dtype/device)
        return:  (B,T,Z,Y,D)  加了 PE
        """
        B,T,Z,Y,D = x_btzyd.shape
        assert T == self.tlen and Z == self.zlen and Y == self.ny and D == self.d_model
        x = x_btzyd.view(B, T*Z*Y, D) + self.pe_flat.unsqueeze(0).to(x_btzyd.device).to(x_btzyd.dtype)
        return x.view(B, T, Z, Y, D)

# ===================== Masks =====================
def causal_mask(L:int, device):
    m = torch.full((L, L), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m

# ===================== Axial 1D blocks =====================
class Axial1D(nn.Module):
    def __init__(self, d_model:int, nhead:int=4, layers:int=1, ffn:int=512, drop:float=0.1, act:str="gelu", causal:bool=False):
        super().__init__()
        enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation=act,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_ly, layers)
        self.causal = causal
    def forward(self, x):  # (B*, L, D)
        if self.causal:
            L = x.size(1)
            m = causal_mask(L, x.device)
            return self.encoder(x, mask=m)
        else:
            return self.encoder(x)

class Axial3DEncoder(nn.Module):
    """
    三轴 1D 注意力编码器（支持交错）。
    - use_interleave=False：原行为，T(全层)→Z(全层)→Y(全层)
    - use_interleave=True & mode='stage'：按轮交错 [T块→Z块→Y块] × cycles
    - use_interleave=True & mode='per_layer'：逐层交错，每轮各轴各走一层
    """
    def __init__(self,
                 d_model:int, tlen:int, zlen:int, ny:int,
                 t_nhead:int=8, t_layers:int=2, t_ffn:int=1024,
                 z_nhead:int=4, z_layers:int=1, z_ffn:int=512,
                 y_nhead:int=4, y_layers:int=1, y_ffn:int=512,
                 drop:float=0.1, act:str="gelu",
                 use_3d_pe:bool=True,
                 use_interleave:bool=False,          # ← 新增：是否交错
                 interleave_mode:str="stage",         # 'stage' or 'per_layer'
                 interleave_cycles:int=2):            # 交错轮数或循环次数
        super().__init__()
        self.d_model = d_model
        self.tlen, self.zlen, self.ny = tlen, zlen, ny
        self.use_3d_pe = bool(use_3d_pe)
        if self.use_3d_pe:
            self.pe3d_flat = FlatPosEnc3D(d_model, tlen, zlen, ny)

        # 把每个轴做成“层列表”，便于交错逐层调用
        def make_stack(nhead, nlayers, ffn):
            enc_ly = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop,
                                                activation=act, batch_first=True, norm_first=True)
            return nn.ModuleList([nn.TransformerEncoder(enc_ly, 1) for _ in range(nlayers)])

        self.t_blocks = make_stack(t_nhead, t_layers, t_ffn)
        self.z_blocks = make_stack(z_nhead, z_layers, z_ffn)
        self.y_blocks = make_stack(y_nhead, y_layers, y_ffn)

        self.use_interleave   = use_interleave
        self.interleave_mode  = interleave_mode
        self.interleave_cycles= int(interleave_cycles)

    def _run_t_once(self, h):
        B,T,Z,Y,D = h.shape
        x = h.permute(0,2,3,1,4).reshape(B*Z*Y, T, D)
        return x, (B,T,Z,Y,D)
    def _back_t(self, x, shape):
        B,T,Z,Y,D = shape
        return x.view(B, Z, Y, T, D).permute(0,3,1,2,4)

    def _run_z_once(self, h):
        B,T,Z,Y,D = h.shape
        x = h.permute(0,1,3,2,4).reshape(B*T*Y, Z, D)
        return x, (B,T,Z,Y,D)
    def _back_z(self, x, shape):
        B,T,Z,Y,D = shape
        return x.view(B, T, Y, Z, D).permute(0,1,3,2,4)

    def _run_y_once(self, h):
        B,T,Z,Y,D = h.shape
        x = h.reshape(B*T*Z, Y, D)
        return x, (B,T,Z,Y,D)
    def _back_y(self, x, shape):
        B,T,Z,Y,D = shape
        return x.view(B, T, Z, Y, D)

    def forward(self, x_proj):   # (B,T,Z,Y,D)
        h = x_proj
        if self.use_3d_pe:
            h = self.pe3d_flat.add_pe(h)

        if not self.use_interleave:
            # === 原行为：T(全层)→Z(全层)→Y(全层) ===
            x, shape = self._run_t_once(h)
            for blk in self.t_blocks: x = blk(x)
            h = self._back_t(x, shape)

            x, shape = self._run_z_once(h)
            for blk in self.z_blocks: x = blk(x)
            h = self._back_z(x, shape)

            x, shape = self._run_y_once(h)
            for blk in self.y_blocks: x = blk(x)
            h = self._back_y(x, shape)
            return h

        # === 交错模式 ===
        if self.interleave_mode == "stage":
            # 按“轮”交错：每轮跑完 T块(全层)→Z块(全层)→Y块(全层)
            for _ in range(self.interleave_cycles):
                x, shape = self._run_t_once(h)
                for blk in self.t_blocks: x = blk(x)
                h = self._back_t(x, shape)

                x, shape = self._run_z_once(h)
                for blk in self.z_blocks: x = blk(x)
                h = self._back_z(x, shape)

                x, shape = self._run_y_once(h)
                for blk in self.y_blocks: x = blk(x)
                h = self._back_y(x, shape)
            return h

        # per_layer：逐层交错
        max_layers = max(len(self.t_blocks), len(self.z_blocks), len(self.y_blocks))
        for i in range(self.interleave_cycles):
            for l in range(max_layers):
                if l < len(self.t_blocks):
                    x, shape = self._run_t_once(h)
                    h = self._back_t(self.t_blocks[l](x), shape)
                if l < len(self.z_blocks):
                    x, shape = self._run_z_once(h)
                    h = self._back_z(self.z_blocks[l](x), shape)
                if l < len(self.y_blocks):
                    x, shape = self._run_y_once(h)
                    h = self._back_y(self.y_blocks[l](x), shape)
        return h

# ===================== 2D PE & Refiner =====================
class PosEnc2D(nn.Module):
    """2D sinusoidal PE for (z,y). 用于 q_pos 与可选的 refine。"""
    def __init__(self, d_model:int, zlen:int, ny:int):
        super().__init__()
        dz = d_model // 2
        dy = d_model - dz
        pez = PosEnc1D.build(dz, zlen)
        pey = PosEnc1D.build(dy, ny)
        self.register_buffer("pez", pez, persistent=True)
        self.register_buffer("pey", pey, persistent=True)
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

# ===================== Model =====================
class AxialSpatiotemporalTransformer(nn.Module):
    """
    编码器：三轴注意力 + 3D PE(flat buffer)
    解码器：learnable q + 2D q_pos
    memory：NO PE（严格按需求）
    记忆池化：--mem-pool {last, mean, all}
    可选：2D refine
    """
    def __init__(self, tlength:int, zlen:int, ny:int, out_c:int=2,
                 d_model:int=256,
                 # encoder axis heads/layers/ffn
                 t_nhead:int=8, t_layers:int=2, t_ffn:int=1024,
                 z_nhead:int=4, z_layers:int=1, z_ffn:int=512,
                 y_nhead:int=4, y_layers:int=1, y_ffn:int=512,
                 # decoder
                 dec_nhead:int=8, dec_layers:int=2, dec_ffn:int=1024,
                 drop:float=0.1, act:str="gelu",
                 use_spatial_refine:bool=False,
                 refine_layers:int=1, refine_nhead:int=4, refine_ffn:int=512, refine_drop:float=0.1, refine_act:str="gelu",
                 use_3d_pe:bool=True,
                 mem_pool:str="last"):
        super().__init__()
        self.tlength, self.zlen, self.ny, self.out_c = tlength, zlen, ny, out_c
        self.d_model = d_model
        self.mem_pool = mem_pool.lower()
        assert self.mem_pool in ("last","mean","all"), "mem_pool 必须是 last/mean/all"

        # 输入通道投影
        self.in_proj = nn.Linear(out_c, d_model)

        # 三轴编码器（仅在编码器端加 3D PE）
        self.axial3d = Axial3DEncoder(
            d_model=d_model, tlen=tlength, zlen=zlen, ny=ny,
            t_nhead=t_nhead, t_layers=t_layers, t_ffn=t_ffn,
            z_nhead=z_nhead, z_layers=z_layers, z_ffn=z_ffn,
            y_nhead=y_nhead, y_layers=y_layers, y_ffn=y_ffn,
            drop=drop, act=act, use_3d_pe=use_3d_pe
        )

        # decoder
        dec_ly = nn.TransformerDecoderLayer(d_model, dec_nhead, dec_ffn, drop, activation=act,
                                            batch_first=False, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_ly, dec_layers)

        # learnable queries + 2D q_pos
        self.q = nn.Parameter(torch.randn(zlen*ny, d_model))
        nn.init.normal_(self.q, std=0.02)
        self.q_pos = PosEnc2D(d_model, zlen, ny)

        # 2D refine
        self.use_spatial_refine = bool(use_spatial_refine)
        if self.use_spatial_refine:
            self.srefine = SpatialRefiner2D(d_model, zlen, ny, nhead=refine_nhead,
                                            num_layers=refine_layers, ffn=refine_ffn,
                                            drop=refine_drop, act=refine_act)
        else:
            self.srefine = None

        self.head = nn.Linear(d_model, out_c)

    def _build_memory(self, h_seq):  # h_seq: (B, T, Z, Y, D)
        B,T,Z,Y,D = h_seq.shape
        if self.mem_pool == "last":
            mem = h_seq[:, -1, :, :, :].reshape(B, Z*Y, D)  # 不加任何 PE
            mem = mem.transpose(0,1)                        # (Z*Y, B, D)
            return mem
        elif self.mem_pool == "mean":
            mem = h_seq.mean(dim=1).reshape(B, Z*Y, D)      # 不加任何 PE
            mem = mem.transpose(0,1)                        # (Z*Y, B, D)
            return mem
        else:  # "all": 使用 T*Z*Y 全部 tokens（不加任何 PE）
            mem = h_seq.reshape(B, T*Z*Y, D)
            mem = mem.transpose(0,1)                        # (T*Z*Y, B, D)
            return mem

    def forward(self, x):  # x: (B, T, Zhist, Ny, C)
        B,T,Z,Y,C = x.shape
        assert Z == self.zlen and Y == self.ny and C == self.out_c

        # 1) 输入通道投影
        h0 = self.in_proj(x)                           # (B,T,Z,Y,D)

        # 2) 三轴编码（仅编码器端加过 3D PE）
        h_seq = self.axial3d(h0)                       # (B,T,Z,Y,D)

        # 3) 组建 memory（decoder 端不再加任何 PE）
        mem = self._build_memory(h_seq)                # (MemLen, B, D)

        # 4) queries (learnable + 2D q_pos)
        q = self.q.unsqueeze(0).expand(B, -1, -1)      # (B, Z*Y, D)
        q = self.q_pos(q)                              # (B, Z*Y, D)
        q = q.transpose(0,1)                           # (Z*Y, B, D)

        # 5) decoder cross-attention
        y = self.decoder(q, mem).transpose(0,1)        # (B, Z*Y, D)

        # 6) 2D refine (可选)
        if self.use_spatial_refine:
            y = self.srefine(y)                        # (B, Z*Y, D)

        # 7) 线性回归头 → (B, Zout, Ny, C)
        y = self.head(y).view(B, self.zlen, self.ny, self.out_c)
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

# ===================== Args & Main =====================
def _parse_target_comps(s: str)->List[int]:
    m={'x':0,'y':1,'z':2}
    s=(s or "yz").lower().strip()
    idx=sorted(set(m[c] for c in s if c in m))
    if not idx: raise ValueError(f"--target-comps '{s}' 无有效通道，示例：y / z / yz / xyz")
    return idx

def parse_args():
    p=argparse.ArgumentParser("Three-axis (T,Z,Y) spatiotemporal Transformer | encoder-only 3D PE (flat buffer)")
    # paths
    p.add_argument("--h5", type=str,
                   default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_interpolation_all_cases.h5")
    p.add_argument("--save-dir", type=str, default="./runs/timeGlobal_axial3D_flatPE")

    # mode
    p.add_argument("--resume", default="")
    p.add_argument("--eval-only", action="store_true")

    # dataset / sampling
    p.add_argument("--tlength", type=int, default=81)
    p.add_argument("--zlength", type=int, default=5)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--x-center-1based", type=int, default=21)
    p.add_argument("--augment", action="store_true")

    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)

    p.add_argument("--samples-per-epoch", type=int, default=0)
    p.add_argument("--val-samples", type=int, default=0)
    p.add_argument("--test-samples", type=int, default=0)

    # normalization
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--target-comps", type=str, default="yz")

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

    # encoder axis heads/layers/ffn
    p.add_argument("--t-nhead", type=int, default=8)
    p.add_argument("--t-layers", type=int, default=2)
    p.add_argument("--t-ffn-dim", type=int, default=1024)

    p.add_argument("--z-nhead", type=int, default=4)
    p.add_argument("--z-layers", type=int, default=1)
    p.add_argument("--z-ffn-dim", type=int, default=512)

    p.add_argument("--y-nhead", type=int, default=4)
    p.add_argument("--y-layers", type=int, default=1)
    p.add_argument("--y-ffn-dim", type=int, default=512)

    # decoder
    p.add_argument("--dec-nhead", type=int, default=8)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--dec-ffn-dim", type=int, default=1024)

    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--act", type=str, default="gelu", choices=["relu","gelu"])

    # 3D PE toggle (encoder only)
    p.add_argument("--no-3d-pe", action="store_true", help="关闭 编码器 3D 位置编码")

    # memory pooling
    p.add_argument("--mem-pool", type=str, default="last", choices=["last","mean","all"],
                   help="last: 取最后一帧; mean: 时间均值; all: 使用 T*Z*Y 全 tokens（decoder 成本×T）")

    # optional 2D refine
    p.add_argument("--use-2d-refine", action="store_true")
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
    p.add_argument("--wandb-project", type=str, default="sec_hist2zfuture_axial3d_flatPE")
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
        z_nhead=args.z_nhead, z_layers=args.z_layers, z_ffn=args.z_ffn_dim,
        y_nhead=args.y_nhead, y_layers=args.y_layers, y_ffn=args.y_ffn_dim,
        dec_nhead=args.dec_nhead, dec_layers=args.dec_layers, dec_ffn=args.dec_ffn_dim,
        drop=args.dropout, act=args.act,
        use_spatial_refine=args.use_2d_refine,
        refine_layers=args.refine_layers, refine_nhead=args.refine_nhead,
        refine_ffn=args.refine_ffn_dim, refine_drop=args.refine_dropout, refine_act=args.refine_act,
        use_3d_pe=(not args.no_3d_pe),
        mem_pool=args.mem_pool
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
