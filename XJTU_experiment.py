"""
主实验脚本 v4: 多模型电池健康预测
（含消融 + 跨域对比 + PatchTST + 迁移策略对比）
=================================================================
Exp1: Fixed 组内 (Batch-1/2)          主对比 + 消融
Exp2: Dynamic 组内 (Batch-3/4)        主对比 + 消融
Exp3: Cross Fixed→Dynamic             主对比 + 迁移策略消融
Exp4: Cross Dynamic→Fixed             主对比 + 迁移策略消融

迁移策略对比（Exp3/Exp4内部，只跑Our模型）：
  1. No-Transfer    : 源域训练，直接测试
  2. Target_Only    : 只用目标域少量数据训练
  3. Domain_Embed   : 域嵌入，不微调
  4. Transfer_FT    : 源域预训练 + 目标域微调（原Transfer）
  5. Transfer_Align : 源域训练加MMD对齐 + 目标域微调
=================================================================
"""

import os
import glob
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from mamba_ssm import Mamba
import warnings
import copy

warnings.filterwarnings('ignore')

# ── 全局字体与样式（放大字体，白色背景）──
plt.rcParams['font.sans-serif']    = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top']    = False
plt.rcParams['axes.spines.right']  = False
plt.rcParams['axes.grid']          = True
plt.rcParams['grid.alpha']         = 0.25
plt.rcParams['grid.linestyle']     = '--'
plt.rcParams['figure.facecolor']   = 'white'   # ← 全局白底
plt.rcParams['axes.facecolor']     = 'white'   # ← 坐标区白底
# ── 放大字体 ──
plt.rcParams['axes.labelsize']     = 16
plt.rcParams['xtick.labelsize']    = 14
plt.rcParams['ytick.labelsize']    = 14
plt.rcParams['axes.titlesize']     = 16
plt.rcParams['legend.fontsize']    = 14

# ── 实验名称简称映射 ──
EXP_SHORT = {
    'Exp1_Fixed':       'Exp1',
    'Exp2_Dynamic':     'Exp2',
    'Exp3_Fixed2Dyn':   'Exp3',
    'Exp4_Dyn2Fixed':   'Exp4',
}

def _exp_short(exp_name):
    """返回实验名称简称"""
    return EXP_SHORT.get(exp_name, exp_name)

# ============================================================
# 全局配置
# ============================================================
TASK_NAMES   = ['tmax', 'soh', 'tmean']
N_OUTPUTS    = 3
SEQ_LEN      = 40
TASK_WEIGHTS = {'tmax': 1.0, 'soh': 1.0, 'tmean': 1.0}

HIGH_TEMP_THRESHOLDS = [35.0, 40.0]
HIGH_TEMP_WEIGHTS    = [1.5,  2.5]

TASK_LABELS = {
    'tmax':  'T$_{max}$ (°C)',
    'soh':   'SOH',
    'tmean': 'T$_{mean}$ (°C)',
}

# ── 主对比模型样式 ──
MODEL_STYLES = {
    'Our':         {'color': '#e74c3c', 'lw': 2.2,
                    'ls': '-',   'zorder': 6},
    'LSTM':        {'color': '#f39c12', 'lw': 1.4,
                    'ls': '--',  'zorder': 5},
    'GRU':         {'color': '#3498db', 'lw': 1.4,
                    'ls': '-.',  'zorder': 5},
    'Transformer': {'color': '#9b59b6', 'lw': 1.4,
                    'ls': ':',   'zorder': 5},
    'CNN_LSTM':    {'color': '#27ae60', 'lw': 1.4,
                    'ls': '--',  'zorder': 5},
    'BiLSTM':      {'color': '#e67e22', 'lw': 1.4,
                    'ls': '-.',  'zorder': 5},
    'PatchTST':    {'color': '#1abc9c', 'lw': 1.4,
                    'ls': ':',   'zorder': 5},
}

# ── 消融模型样式 ──
ABLATION_STYLES = {
    'Our':           {'color': '#e74c3c', 'lw': 2.2,
                      'ls': '-',  'zorder': 6},
    'Our_noSE':      {'color': '#f39c12', 'lw': 1.4,
                      'ls': '--', 'zorder': 5},
    'Our_TCNonly':   {'color': '#3498db', 'lw': 1.4,
                      'ls': '-.',  'zorder': 5},
    'Our_Mambaonly': {'color': '#9b59b6', 'lw': 1.4,
                      'ls': ':',  'zorder': 5},
    'Our_noGate':    {'color': '#27ae60', 'lw': 1.4,
                      'ls': '--', 'zorder': 5},
    'Our_noHTW':     {'color': '#e67e22', 'lw': 1.4,
                      'ls': '-.',  'zorder': 5},
}

# ── 迁移策略对比样式（Exp3/Exp4 内部）──
TRANSFER_STYLES = {
    'No-Transfer':    {'color': '#7f8c8d', 'lw': 1.4,
                       'ls': ':',   'zorder': 4},
    'Target_Only':    {'color': '#2980b9', 'lw': 1.4,
                       'ls': '--',  'zorder': 4},
    'Domain_Embed':   {'color': '#8e44ad', 'lw': 1.4,
                       'ls': '-.',  'zorder': 5},
    'Transfer_FT':    {'color': '#f39c12', 'lw': 1.6,
                       'ls': '--',  'zorder': 5},
    'Transfer_Align': {'color': '#e74c3c', 'lw': 2.2,
                       'ls': '-',   'zorder': 6},
}

PALETTE = {
    'measured': '#95a5a6',
    'bg_even':  '#ffffff',   # ← 改为白色
    'bg_odd':   '#ffffff',   # ← 改为白色
}

# 主对比模型列表
MAIN_MODEL_TYPES_KEYS = [
    'LSTM', 'GRU', 'Transformer',
    'CNN_LSTM', 'BiLSTM', 'PatchTST', 'Our'
]

# 消融模型列表
ABLATION_MODEL_KEYS = [
    'Our_noSE', 'Our_TCNonly', 'Our_Mambaonly',
    'Our_noGate', 'Our_noHTW', 'Our'
]

# 迁移策略列表
TRANSFER_STRATEGY_KEYS = [
    'No-Transfer', 'Target_Only', 'Domain_Embed',
    'Transfer_FT', 'Transfer_Align'
]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)


# ============================================================
# 1. 模型定义
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.GELU(),
            nn.Linear(ch // r, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean(1)).unsqueeze(1)


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        pad = (3 - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, 3, padding=pad,
                              dilation=dilation, groups=ch)
        self.pw   = nn.Conv1d(ch, ch, 1)
        self.norm = nn.BatchNorm1d(ch)
        self.act  = nn.GELU()
        self.pad  = pad
    def forward(self, x):
        res = x
        o   = self.conv(x)
        if self.pad > 0:
            o = o[:, :, :-self.pad]
        return self.act(self.norm(self.pw(o) + res))


def _make_heads(d_model):
    return nn.ModuleDict({
        'tmax':  nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1)),
        'soh':   nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1)),
        'tmean': nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1)),
        'delta': nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Softplus()),
    })


def _forward_heads(heads, h, x, domain_emb=None,
                   domain_id=None):
    if domain_emb is not None and domain_id is not None:
        h = h + domain_emb(domain_id)
    return (heads['tmax'](h),  heads['soh'](h),
            heads['tmean'](h), heads['delta'](h))


# ─────────────────────────────────────────────
# 主模型：支持 return_feat 用于 MMD 对齐
# ─────────────────────────────────────────────

class OurModel(nn.Module):
    def __init__(self, input_dim, d_model=128,
                 use_domain=False):
        super().__init__()
        self.use_domain = use_domain
        if use_domain:
            self.domain_emb = nn.Embedding(2, d_model)
        else:
            self.domain_emb = None
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.se           = SEBlock(d_model)
        self.tcn_stream   = nn.Sequential(
            TCNBlock(d_model, 1), TCNBlock(d_model, 2),
            nn.Dropout(0.05))
        self.mamba_stream = Mamba(
            d_model=d_model, d_state=128,
            d_conv=4, expand=2)
        self.gate   = nn.Sequential(
            nn.Linear(d_model*2, d_model*2), nn.Sigmoid())
        self.fusion = nn.Linear(d_model*2, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.heads  = _make_heads(d_model)

    def forward(self, x, domain_id=None,
                return_feat=False):
        h = self.se(self.input_proj(x))
        t = self.tcn_stream(
            h.permute(0,2,1)).permute(0,2,1)
        m = self.mamba_stream(h)
        c = torch.cat([t, m], -1)
        f = self.fusion(c * self.gate(c))
        feat = self.norm(f)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat, x,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=8,
                 use_domain=False):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim,
                               batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(hidden_dim, N_OUTPUTS)
    def forward(self, x, domain_id=None,
                return_feat=False):
        _, (h, _) = self.lstm(x)
        feat = self.dropout(h[-1])
        o    = self.fc(feat)
        z    = torch.zeros(o.shape[0], 1, device=x.device)
        out  = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=8,
                 use_domain=False):
        super().__init__()
        self.gru     = nn.GRU(input_dim, hidden_dim,
                              batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(hidden_dim, N_OUTPUTS)
    def forward(self, x, domain_id=None,
                return_feat=False):
        _, h   = self.gru(x)
        feat   = self.dropout(h[-1])
        o      = self.fc(feat)
        z      = torch.zeros(o.shape[0], 1, device=x.device)
        out    = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=16,
                 use_domain=False):
        super().__init__()
        self.emb = nn.Linear(input_dim, d_model)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4,
            batch_first=True, dropout=0.6)
        self.tf = nn.TransformerEncoder(el, num_layers=1)
        self.fc = nn.Linear(d_model, N_OUTPUTS)
    def forward(self, x, domain_id=None,
                return_feat=False):
        feat = self.tf(self.emb(x))[:, -1, :]
        o    = self.fc(feat)
        z    = torch.zeros(o.shape[0], 1, device=x.device)
        out  = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=8,
                 use_domain=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.6))
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim,
                               batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_dim, N_OUTPUTS)

    def forward(self, x, domain_id=None,
                return_feat=False):
        c = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (h, _) = self.lstm(c)
        feat = self.dropout(h[-1])
        o    = self.fc(feat)
        z    = torch.zeros(o.shape[0], 1, device=x.device)
        out  = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16,
                 use_domain=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.6)
        self.fc      = nn.Linear(hidden_dim * 2, N_OUTPUTS)

    def forward(self, x, domain_id=None,
                return_feat=False):
        _, (h, _) = self.lstm(x)
        feat = self.dropout(
            torch.cat([h[-2], h[-1]], dim=-1))
        o   = self.fc(feat)
        z   = torch.zeros(o.shape[0], 1, device=x.device)
        out = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


class PatchTSTModel(nn.Module):
    def __init__(self, input_dim,
                 patch_len=16, stride=4,
                 d_model=8, nhead=4,
                 num_layers=1, dropout=0.6,
                 use_domain=False):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (SEQ_LEN - patch_len) // stride + 1
        patch_dim      = patch_len * input_dim

        self.patch_emb = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model))
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            batch_first=True, dropout=dropout,
            dim_feedforward=d_model * 4)
        self.encoder = nn.TransformerEncoder(
            el, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, N_OUTPUTS)

    def forward(self, x, domain_id=None,
                return_feat=False):
        B, T, F = x.shape
        patches  = []
        for i in range(self.n_patches):
            start = i * self.stride
            end   = start + self.patch_len
            if end > T:
                break
            patches.append(
                x[:, start:end, :].reshape(B, -1))
        patches = torch.stack(patches, dim=1)
        h = self.patch_emb(patches) + \
            self.pos_emb[:, :patches.shape[1], :]
        h    = self.encoder(h)
        h    = self.norm(h)
        feat = self.dropout(h.mean(dim=1))
        o    = self.fc(feat)
        z    = torch.zeros(o.shape[0], 1, device=x.device)
        out  = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


# ─────────────────────────────────────────────
# 消融变体
# ─────────────────────────────────────────────

class OurModel_noSE(nn.Module):
    def __init__(self, input_dim, d_model=128,
                 use_domain=False):
        super().__init__()
        self.use_domain = use_domain
        self.domain_emb = (nn.Embedding(2, d_model)
                           if use_domain else None)
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.tcn_stream   = nn.Sequential(
            TCNBlock(d_model, 1), TCNBlock(d_model, 2),
            nn.Dropout(0.05))
        self.mamba_stream = Mamba(
            d_model=d_model, d_state=128,
            d_conv=4, expand=2)
        self.gate   = nn.Sequential(
            nn.Linear(d_model*2, d_model*2), nn.Sigmoid())
        self.fusion = nn.Linear(d_model*2, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.heads  = _make_heads(d_model)

    def forward(self, self_x, domain_id=None,
                return_feat=False):
        h = self.input_proj(self_x)
        t = self.tcn_stream(
            h.permute(0,2,1)).permute(0,2,1)
        m = self.mamba_stream(h)
        c = torch.cat([t, m], -1)
        f = self.fusion(c * self.gate(c))
        feat = self.norm(f)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat, self_x,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class OurModel_TCNonly(nn.Module):
    def __init__(self, input_dim, d_model=64,
                 use_domain=False):
        super().__init__()
        self.use_domain = use_domain
        self.domain_emb = (nn.Embedding(2, d_model)
                           if use_domain else None)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.se         = SEBlock(d_model)
        self.tcn_stream = nn.Sequential(
            TCNBlock(d_model, 1), TCNBlock(d_model, 2),
            nn.Dropout(0.05))
        self.norm  = nn.LayerNorm(d_model)
        self.heads = _make_heads(d_model)

    def forward(self, x, domain_id=None,
                return_feat=False):
        h = self.se(self.input_proj(x))
        h = self.tcn_stream(
            h.permute(0,2,1)).permute(0,2,1)
        feat = self.norm(h)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat, x,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class OurModel_Mambaonly(nn.Module):
    def __init__(self, input_dim, d_model=128,
                 use_domain=False):
        super().__init__()
        self.use_domain = use_domain
        self.domain_emb = (nn.Embedding(2, d_model)
                           if use_domain else None)
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.se           = SEBlock(d_model)
        self.mamba_stream = Mamba(
            d_model=d_model, d_state=128,
            d_conv=4, expand=2)
        self.norm  = nn.LayerNorm(d_model)
        self.heads = _make_heads(d_model)

    def forward(self, x, domain_id=None,
                return_feat=False):
        h = self.se(self.input_proj(x))
        h = self.mamba_stream(h)
        feat = self.norm(h)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat, x,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class OurModel_noGate(nn.Module):
    def __init__(self, input_dim, d_model=64,
                 use_domain=False):
        super().__init__()
        self.use_domain = use_domain
        self.domain_emb = (nn.Embedding(2, d_model)
                           if use_domain else None)
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.se           = SEBlock(d_model)
        self.tcn_stream   = nn.Sequential(
            TCNBlock(d_model, 1), TCNBlock(d_model, 2),
            nn.Dropout(0.05))
        self.mamba_stream = Mamba(
            d_model=d_model, d_state=128,
            d_conv=4, expand=2)
        self.norm  = nn.LayerNorm(d_model)
        self.heads = _make_heads(d_model)

    def forward(self, x, domain_id=None,
                return_feat=False):
        h = self.se(self.input_proj(x))
        t = self.tcn_stream(
            h.permute(0,2,1)).permute(0,2,1)
        m = self.mamba_stream(h)
        f    = 0.5 * t + 0.5 * m
        feat = self.norm(f)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat, x,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class OurModel_noHTW(nn.Module):
    def __init__(self, input_dim, d_model=64,
                 use_domain=False):
        super().__init__()
        self._m = OurModel(input_dim, d_model, use_domain)

    def forward(self, x, domain_id=None,
                return_feat=False):
        return self._m(x, domain_id, return_feat)


# ── 模型注册表 ──
MAIN_MODEL_CLASSES = {
    'LSTM':        LSTMModel,
    'GRU':         GRUModel,
    'Transformer': TransformerModel,
    'CNN_LSTM':    CNNLSTMModel,
    'BiLSTM':      BiLSTMModel,
    'PatchTST':    PatchTSTModel,
    'Our':         OurModel,
}

ABLATION_MODEL_CLASSES = {
    'Our_noSE':      OurModel_noSE,
    'Our_TCNonly':   OurModel_TCNonly,
    'Our_Mambaonly': OurModel_Mambaonly,
    'Our_noGate':    OurModel_noGate,
    'Our_noHTW':     OurModel_noHTW,
    'Our':           OurModel,
}

NO_HTW_MODELS = {'Our_noHTW'}


# ============================================================
# 2. 数据加载
# ============================================================

def extract_batteries(pkl_files, prefix="",
                      domain_id=0,
                      exclude_features=None):
    bats   = []
    is_dyn = "Dyn" in prefix
    for f in pkl_files:
        df = pd.read_pickle(f).ffill().bfill()
        col_t = next((c for c in df.columns
                      if 'Target_T_max' in str(c)), None)
        col_s = next((c for c in df.columns
                      if 'Target_SOH' in str(c)), None)
        col_m = next((c for c in df.columns
                      if 'Target_T_mean' in str(c)), None)
        col_c = next((c for c in df.columns
                      if 'cycle' in str(c).lower()), None)
        if col_t is None or col_c is None:
            continue
        f_cols = [c for c in df.columns
                  if c not in [col_c, 'battery_id']
                  and 'Target_' not in str(c)]
        if exclude_features:
            f_cols = [c for c in f_cols
                      if c not in exclude_features]
        splits = np.where(
            np.diff(df[col_c].values) < 0)[0] + 1
        for si, sub in enumerate(np.split(df, splits)):
            sub = sub.copy()
            if len(sub) < 60:
                continue
            for c in f_cols:
                sub[c] = gaussian_filter1d(
                    sub[c].rolling(7, 1, center=True)
                    .median(), sigma=1.0)
            sig = 2.5 if is_dyn else 0.8
            ta  = (gaussian_filter1d(
                sub[col_t].values, sig)
                   if col_t else np.zeros(len(sub)))
            sa  = (gaussian_filter1d(
                sub[col_s].values, 0.5)
                   if col_s else np.zeros(len(sub)))
            ma  = (gaussian_filter1d(
                sub[col_m].values, sig)
                   if col_m else np.zeros(len(sub)))
            bats.append({
                'id':        (f"{prefix}_"
                              f"{os.path.basename(f).split('.')[0]}"
                              f"_B{si+1}"),
                'feat':      sub[f_cols].values,
                'tmax':      ta,
                'soh':       sa,
                'tmean':     ma,
                'delta_t':   ta - ma,
                'domain_id': domain_id,
                'feat_cols': f_cols,
            })
    print(f"  [{prefix}] {len(bats)} batteries loaded")
    return bats


# ============================================================
# 3. Dataset
# ============================================================

class BatteryDataset(Dataset):
    def __init__(self, b_list, sx=None, sy_t=None,
                 sy_s=None, sy_m=None, sy_d=None):
        if sx is None:
            self.sx   = StandardScaler().fit(
                np.vstack([b['feat'] for b in b_list]))
            self.sy_t = MinMaxScaler().fit(
                np.concatenate(
                    [b['tmax'] for b in b_list]
                ).reshape(-1,1))
            self.sy_s = MinMaxScaler().fit(
                np.concatenate(
                    [b['soh'] for b in b_list]
                ).reshape(-1,1))
            self.sy_m = MinMaxScaler().fit(
                np.concatenate(
                    [b['tmean'] for b in b_list]
                ).reshape(-1,1))
            self.sy_d = MinMaxScaler().fit(
                np.concatenate(
                    [b['delta_t'] for b in b_list]
                ).reshape(-1,1))
        else:
            self.sx   = sx;   self.sy_t = sy_t
            self.sy_s = sy_s; self.sy_m = sy_m
            self.sy_d = sy_d

        self.x, self.y, self.did, self.traw = [], [], [], []
        for b in b_list:
            fs = np.clip(
                self.sx.transform(b['feat']), -3.5, 3.5)
            tn = self.sy_t.transform(
                b['tmax'].reshape(-1,1)).flatten()
            sn = self.sy_s.transform(
                b['soh'].reshape(-1,1)).flatten()
            mn = self.sy_m.transform(
                b['tmean'].reshape(-1,1)).flatten()
            dn = self.sy_d.transform(
                b['delta_t'].reshape(-1,1)).flatten()
            did = b.get('domain_id', 0)
            N   = len(fs)
            for i in range(N - SEQ_LEN):
                j = i + SEQ_LEN
                self.x.append(fs[i:j])
                self.y.append([tn[j], sn[j], mn[j], dn[j]])
                self.did.append(did)
                self.traw.append(b['tmax'][j])

        self.x    = np.array(self.x,    dtype=np.float32)
        self.y    = np.array(self.y,    dtype=np.float32)
        self.did  = np.array(self.did,  dtype=np.int64)
        self.traw = np.array(self.traw, dtype=np.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]),
                torch.from_numpy(self.y[i]),
                torch.tensor(self.did[i]),
                torch.tensor(self.traw[i]))
    @property
    def feat_dim(self): return self.x.shape[2]
    def get_scalers(self):
        return (self.sx, self.sy_t, self.sy_s,
                self.sy_m, self.sy_d)


# ============================================================
# 4. 损失 + 训练
# ============================================================

def get_temp_weights(tmax_raw, device):
    w = torch.ones_like(tmax_raw)
    for th, wt in zip(HIGH_TEMP_THRESHOLDS,
                      HIGH_TEMP_WEIGHTS):
        w = torch.where(tmax_raw >= th,
                        torch.tensor(wt, device=device), w)
    return w


def compute_loss(pt, ps, pm, pd,
                 gt, gs, gm, gd,
                 traw, weights, dw, htw, device):
    def wh(pred, target, sw=None):
        base = nn.functional.huber_loss(
            pred.squeeze(), target.squeeze(),
            reduction='none')
        if sw is not None:
            base = base * sw
        return base.mean()
    tw = (get_temp_weights(traw, device) if htw
          else torch.ones(traw.shape[0], device=device))
    return (weights[0] * wh(pt, gt, tw) +
            weights[1] * wh(ps, gs) +
            weights[2] * wh(pm, gm, tw) +
            dw * wh(pd, gd))


def mmd_loss(src_feat, tgt_feat, kernel='rbf'):
    """Maximum Mean Discrepancy (MMD) 损失"""
    def rbf_kernel(x, y, bandwidth):
        xx = (x ** 2).sum(1, keepdim=True)
        yy = (y ** 2).sum(1, keepdim=True)
        xy = x @ y.T
        dist = xx + yy.T - 2 * xy
        return torch.exp(-dist / (2 * bandwidth ** 2))

    bandwidths = [0.5, 1.0, 2.0, 4.0]
    mmd = torch.tensor(0.0, device=src_feat.device)
    n_s = src_feat.shape[0]
    n_t = tgt_feat.shape[0]

    for bw in bandwidths:
        k_ss = rbf_kernel(src_feat, src_feat, bw).sum() \
               / (n_s * n_s)
        k_tt = rbf_kernel(tgt_feat, tgt_feat, bw).sum() \
               / (n_t * n_t)
        k_st = rbf_kernel(src_feat, tgt_feat, bw).sum() \
               / (n_s * n_t)
        mmd  = mmd + k_ss + k_tt - 2 * k_st

    return mmd / len(bandwidths)


def train_engine(model, loader, device,
                 epochs=60, lr=1e-3,
                 weights=(1.,1.,1.),
                 dw=0.3, htw=True, ud=False,
                 align_loader=None,
                 mmd_weight=0.1):
    opt = optim.AdamW(model.parameters(),
                      lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs)

    use_align = (align_loader is not None)
    tgt_iter  = iter(align_loader) if use_align else None

    for _ in range(epochs):
        model.train()
        for xb, yb, did, traw in loader:
            xb   = xb.to(device)
            yb   = yb.to(device)
            did  = did.to(device)
            traw = traw.to(device)
            opt.zero_grad()

            if use_align:
                try:
                    tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter  = iter(align_loader)
                    tgt_batch = next(tgt_iter)
                xt = tgt_batch[0].to(device)

                (pt, ps, pm, pd), src_feat = model(
                    xb, did if ud else None,
                    return_feat=True)

                with torch.no_grad():
                    tgt_did = torch.ones(
                        xt.shape[0], dtype=torch.long,
                        device=device)
                _, tgt_feat = model(
                    xt, tgt_did if ud else None,
                    return_feat=True)

                task_loss = compute_loss(
                    pt, ps, pm, pd,
                    yb[:,0:1], yb[:,1:2],
                    yb[:,2:3], yb[:,3:4],
                    traw, weights, dw, htw, device)

                align_loss = mmd_loss(src_feat, tgt_feat)
                loss = task_loss + mmd_weight * align_loss

            else:
                pt, ps, pm, pd = model(
                    xb, did if ud else None)
                loss = compute_loss(
                    pt, ps, pm, pd,
                    yb[:,0:1], yb[:,1:2],
                    yb[:,2:3], yb[:,3:4],
                    traw, weights, dw, htw, device)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


# ============================================================
# 5. 评估
# ============================================================

def evaluate_battery(model, bat, ds, device,
                     sigma=1.0, ud=False):
    sx, sy_t, sy_s, sy_m, sy_d = ds.get_scalers()
    tds = BatteryDataset(
        [bat], sx, sy_t, sy_s, sy_m, sy_d)
    ldr = DataLoader(tds, 256, shuffle=False)
    pt_l, ps_l, pm_l, pd_l = [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, _, did, _ in ldr:
            xb  = xb.to(device)
            did = did.to(device)
            pt, ps, pm, pd = model(
                xb, did if ud else None)
            pt_l.extend(pt.cpu().numpy().flatten())
            ps_l.extend(ps.cpu().numpy().flatten())
            pm_l.extend(pm.cpu().numpy().flatten())
            pd_l.extend(pd.cpu().numpy().flatten())

    L      = SEQ_LEN
    result = {}
    for task, sy, preds in zip(
            TASK_NAMES, [sy_t, sy_s, sy_m],
            [pt_l, ps_l, pm_l]):
        ts   = bat[task][L:]
        pred = sy.inverse_transform(
            np.array(preds).reshape(-1,1)).flatten()
        pred = gaussian_filter1d(pred, sigma)
        n    = min(len(ts), len(pred))
        ts, pred = ts[:n], pred[:n]
        result[task] = {
            'mae':  mean_absolute_error(ts, pred),
            'rmse': np.sqrt(mean_squared_error(ts, pred)),
            'ts':   ts,
            'ps':   pred,
        }

    dps = sy_d.inverse_transform(
        np.array(pd_l).reshape(-1,1)).flatten()
    dps = gaussian_filter1d(dps, sigma)
    dts = bat['delta_t'][L:]
    n   = min(len(dts), len(dps))
    result['delta_t'] = {
        'mae':    mean_absolute_error(dts[:n], dps[:n]),
        'rmse':   np.sqrt(mean_squared_error(
            dts[:n], dps[:n])),
        'ts':     dts[:n],
        'ps':     dps[:n],
        'values': dps[:n],
    }

    tt, tp = result['tmax']['ts'], result['tmax']['ps']
    hi = tt >= HIGH_TEMP_THRESHOLDS[0]
    result['tmax_high'] = {
        'mae':   (mean_absolute_error(tt[hi], tp[hi])
                  if hi.sum() > 0 else np.nan),
        'count': int(hi.sum()),
    }
    return result


# ============================================================
# 6. 保存函数
# ============================================================

def _short_id(bid):
    parts = str(bid).split('_')
    return '_'.join(parts[-2:]) if len(parts) >= 2 else bid


def save_test_metrics(exp_name, results, save_dir,
                      suffix=''):
    rows = []
    for mn, bat_results in results.items():
        for r in bat_results:
            row = {
                'Experiment': exp_name,
                'Battery':    r['id'],
                'Model':      mn,
            }
            for task in TASK_NAMES:
                row[f'MAE_{task}']  = r[task]['mae']
                row[f'RMSE_{task}'] = r[task]['rmse']
            rows.append(row)
    if rows:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{exp_name}{suffix}_test_metrics.csv"
        pd.DataFrame(rows).to_csv(
            os.path.join(save_dir, fname),
            index=False, float_format='%.6f')
        print(f"  [Metrics] → {fname}")
    return rows


def save_prediction_data(exp_name, results, save_dir,
                         top_ids=None):
    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    if top_ids is None:
        first_key = next(iter(results))
        top_ids   = {r['id'] for r in results[first_key]}

    all_rows    = []
    metric_rows = []

    for mn, bat_results in results.items():
        model_rows = []
        for r in bat_results:
            bat_id = r['id']
            if bat_id not in top_ids:
                continue
            N = len(r['tmax']['ts'])
            for i in range(N):
                row = {
                    'bat_id':       bat_id,
                    'model':        mn,
                    'exp':          exp_name,
                    'step':         i,
                    'tmax_true':    r['tmax']['ts'][i]        if i < len(r['tmax']['ts'])        else np.nan,
                    'tmax_pred':    r['tmax']['ps'][i]        if i < len(r['tmax']['ps'])        else np.nan,
                    'soh_true':     r['soh']['ts'][i]         if i < len(r['soh']['ts'])         else np.nan,
                    'soh_pred':     r['soh']['ps'][i]         if i < len(r['soh']['ps'])         else np.nan,
                    'tmean_true':   r['tmean']['ts'][i]       if i < len(r['tmean']['ts'])       else np.nan,
                    'tmean_pred':   r['tmean']['ps'][i]       if i < len(r['tmean']['ps'])       else np.nan,
                    'delta_t_true': r['delta_t']['ts'][i]     if i < len(r['delta_t']['ts'])     else np.nan,
                    'delta_t_pred': r['delta_t']['values'][i] if i < len(r['delta_t']['values']) else np.nan,
                }
                model_rows.append(row)
                all_rows.append(row)

            for task in TASK_NAMES:
                metric_rows.append({
                    'exp': exp_name, 'model': mn,
                    'bat_id': bat_id, 'task': task,
                    'mae':      r[task]['mae'],
                    'rmse':     r[task]['rmse'],
                    'n_points': len(r[task]['ts']),
                })
            metric_rows.append({
                'exp': exp_name, 'model': mn,
                'bat_id': bat_id, 'task': 'tmax_high',
                'mae':      r['tmax_high']['mae'],
                'rmse':     np.nan,
                'n_points': r['tmax_high']['count'],
            })

        if model_rows:
            path = os.path.join(
                pred_dir,
                f"{exp_name}_{mn}_predictions.csv")
            pd.DataFrame(model_rows).to_csv(
                path, index=False, float_format='%.6f')

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(pred_dir,
                         f"{exp_name}_predictions.csv"),
            index=False, float_format='%.6f')

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(
            os.path.join(pred_dir,
                         f"{exp_name}_metrics.csv"),
            index=False, float_format='%.6f')

    print(f"  [Save] Predictions → "
          f"{exp_name} ({len(top_ids)} batteries)")
    return pred_dir


# ============================================================
# 7. 可视化 ── 核心修改区域
# ============================================================

def _make_metric_text(mae, rmse):
    return f"MAE={mae:.4f}\nRMSE={rmse:.4f}"


def _shade_error(ax, x, ts, ps, color, alpha=0.12):
    ax.fill_between(x, ts, ps,
                    where=(ps >= ts), interpolate=True,
                    alpha=alpha, color=color)
    ax.fill_between(x, ts, ps,
                    where=(ps < ts), interpolate=True,
                    alpha=alpha, color=color)


def plot_predictions(exp_name, results, save_dir,
                     n_show=4, styles=None,
                     tag='', is_cross=False):
    """
    修改要点：
      1. 字体放大（subplot 内字体统一放大）
      2. 底色全部改为白色
      3. 去掉大标题（suptitle），每列仅显示电池ID
      4. 实验名称使用简称
      5. dpi=300 提高清晰度
    """
    if styles is None:
        styles = MODEL_STYLES

    model_keys = [k for k in styles if k in results]
    if not model_keys:
        return

    ref_key = ('Transfer_Align' if 'Transfer_Align' in results
               else ('Our' if 'Our' in results
                     else model_keys[0]))

    scores = [{
        'id':    r['id'],
        'score': r['tmax']['mae'] + r['tmax']['rmse'],
    } for r in results[ref_key]]
    sorted_b = sorted(scores, key=lambda x: x['score'])
    ids      = [x['id'] for x in sorted_b[:n_show]]

    n_tasks = len(TASK_NAMES)

    # ── 每个子图宽度加大，高度加大，留出图例空间 ──
    fig_w = 5.5 * n_show + 1.2
    fig_h = 4.2 * n_tasks + 1.8   # 底部留图例

    fig = plt.figure(figsize=(fig_w, fig_h),
                     facecolor='white')          # ← 白底
    gs  = gridspec.GridSpec(
        n_tasks, n_show, figure=fig,
        left=0.07, right=0.97,
        top=0.97,  bottom=0.12,   # bottom 留图例
        hspace=0.45, wspace=0.30)

    # ── 图例句柄 ──
    legend_handles = [
        Line2D([0], [0], color=PALETTE['measured'],
               lw=2.5, ls='-', label='Measured')]
    for mn in model_keys:
        st = styles[mn]
        legend_handles.append(
            Line2D([0], [0], color=st['color'],
                   lw=st['lw'], ls=st['ls'],
                   label=mn.replace('_', '-')))

    for ci, bid in enumerate(ids):
        ref_r = next((r for r in results[ref_key]
                      if r['id'] == bid), None)
        if ref_r is None:
            continue

        for ri, task in enumerate(TASK_NAMES):
            ax = fig.add_subplot(gs[ri, ci])
            ax.set_facecolor('white')            # ← 白底

            ts = ref_r[task]['ts']
            N  = len(ts)
            x  = np.arange(N)

            _shade_error(ax, x, ts, ref_r[task]['ps'],
                         styles[ref_key]['color'],
                         alpha=0.10)
            ax.plot(x, ts, color=PALETTE['measured'],
                    lw=1.8, ls='-', alpha=0.65, zorder=10)

            for mn in model_keys:
                r_m = next((r for r in results[mn]
                            if r['id'] == bid), None)
                if r_m is None:
                    continue
                ps_m = r_m[task]['ps']
                n_   = min(N, len(ps_m))
                st   = styles[mn]
                ax.plot(x[:n_], ps_m[:n_],
                        color=st['color'],
                        lw=st['lw'], ls=st['ls'],
                        alpha=0.90, zorder=st['zorder'])

            mae_  = ref_r[task]['mae']
            rmse_ = ref_r[task]['rmse']
            # ── 指标文字框放大 ──
            ax.text(0.98, 0.97,
                    _make_metric_text(mae_, rmse_),
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight='bold',              # ← 放大
                    va='top', ha='right',
                    color='#000000',
                    bbox=dict(boxstyle='round,pad=0.35',
                              fc='white', ec='#000000',
                              alpha=0.90))

            # ── 列标题：只显示电池 ID，无实验名/无标题 ──
            if ri == 0:
                ax.set_title(
                    _short_id(bid),
                    fontsize=13,              # ← 放大
                    fontweight='bold',
                    pad=5, color='#1a1a2e')

            # ── Y 轴标签 ──
            if ci == 0:
                ax.set_ylabel(
                    TASK_LABELS[task],
                    fontsize=13,              # ← 放大
                    labelpad=5)

            # ── X 轴标签 ──
            if ri == n_tasks - 1:
                ax.set_xlabel('Cycle', fontsize=14)  # ← 放大

            # ── 刻度字体放大 ──
            ax.tick_params(axis='both', labelsize=13)  # ← 放大
            ax.yaxis.set_major_locator(
                plt.MaxNLocator(nbins=4, prune='both'))
            ax.xaxis.set_major_locator(
                plt.MaxNLocator(nbins=5, integer=True))


    # ── 图例放底部，字体放大 ──
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(len(legend_handles), 9),
        fontsize=15,                          # ← 放大
        frameon=True, framealpha=0.92,
        edgecolor='#cccccc',
        bbox_to_anchor=(0.50, 0.01))

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir,
                       f"{exp_name}{tag}_pred.png")
    # ── dpi=300 提高清晰度 ──
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  [Plot] {os.path.basename(out)}")


# ── 其余可视化函数（heatmap 等）保持白底 + 高清 ──

def plot_transfer_strategy_heatmap(
        transfer_metrics, save_dir):
    df = pd.DataFrame(transfer_metrics)
    if df.empty:
        return

    for exp_name in df['Experiment'].unique():
        sub = df[df['Experiment'] == exp_name]
        strategies = TRANSFER_STRATEGY_KEYS
        tasks      = TASK_NAMES

        mat = np.full(
            (len(strategies), len(tasks)), np.nan)
        for si, strat in enumerate(strategies):
            for ti, task in enumerate(tasks):
                row = sub[(sub['Model'] == strat) &
                          (sub['Task'] == task)]
                if not row.empty:
                    mat[si, ti] = row['MAE'].values[0]

        fig, ax = plt.subplots(
            figsize=(5, 0.7*len(strategies)+1.5),
            facecolor='white')
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
        im   = ax.imshow(mat, aspect='auto',
                         cmap='RdYlGn_r',
                         vmin=vmin, vmax=vmax)
        for si in range(len(strategies)):
            for ti in range(len(tasks)):
                v = mat[si, ti]
                if not np.isnan(v):
                    fc = ('white' if v > (vmin+vmax)/2
                          else '#2c3e50')
                    ax.text(ti, si, f"{v:.4f}",
                            ha='center', va='center',
                            fontsize=9,
                            fontweight='bold', color=fc)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(
            [TASK_LABELS[t] for t in tasks],
            fontsize=9)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(
            [s.replace('_', '\n') for s in strategies],
            fontsize=8)
        exp_s = _exp_short(exp_name)
        ax.set_title(
            f"{exp_s} — Transfer Strategy MAE",
            fontsize=11, fontweight='bold',
            color='#1a1a2e', pad=8)
        plt.colorbar(im, ax=ax,
                     fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = os.path.join(
            save_dir,
            f"{exp_name}_Transfer_Heatmap.png")
        plt.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  [Plot] {os.path.basename(out)}")


def plot_overview_metrics(all_metrics, save_dir):
    df = pd.DataFrame(all_metrics)
    df = df[df['Task'].isin(TASK_NAMES)].copy()

    exps   = df['Experiment'].unique()
    models = list(MAIN_MODEL_CLASSES.keys())
    tasks  = TASK_NAMES
    n_exp  = len(exps)

    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(5*len(tasks), 1.2*n_exp+2),
        facecolor='white')
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        mat = np.full((n_exp, len(models)), np.nan)
        for ei, exp in enumerate(exps):
            for mi, mn in enumerate(models):
                sub = df[(df['Experiment'] == exp) &
                         (df['Model'] == mn) &
                         (df['Task'] == task)]
                if not sub.empty:
                    mat[ei, mi] = sub['MAE'].values[0]
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
        im   = ax.imshow(mat, aspect='auto',
                         cmap='RdYlGn_r',
                         vmin=vmin, vmax=vmax)
        for ei in range(n_exp):
            for mi in range(len(models)):
                v = mat[ei, mi]
                if not np.isnan(v):
                    fc = ('white' if v > (vmin+vmax)/2
                          else '#2c3e50')
                    ax.text(mi, ei, f"{v:.4f}",
                            ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color=fc)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            [m.replace('_', '-') for m in models],
            fontsize=8, rotation=30, ha='right')
        ax.set_yticks(range(n_exp))
        # ── Y 轴实验名使用简称 ──
        ax.set_yticklabels(
            [_exp_short(e) for e in exps],
            fontsize=9)
        ax.set_title(
            f"{TASK_LABELS[task]}\n(MAE)",
            fontsize=10, fontweight='bold',
            color='#2c3e50', pad=8)
        plt.colorbar(im, ax=ax,
                     fraction=0.046, pad=0.04)

    fig.suptitle(
        "All Experiments — MAE Overview",
        fontsize=13, fontweight='bold',
        y=1.02, color='#1a1a2e')
    plt.tight_layout()
    out = os.path.join(
        save_dir, "Overview_MAE_Heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  [Plot] Overview heatmap saved.")


def plot_ablation_heatmap(abl_metrics, save_dir):
    df = pd.DataFrame(abl_metrics)
    if df.empty:
        return

    exps   = df['Experiment'].unique()
    models = ABLATION_MODEL_KEYS
    tasks  = TASK_NAMES
    n_exp  = len(exps)

    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(4.5*len(tasks),
                 1.2*n_exp*len(models)/2+2),
        facecolor='white')
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        row_labels = []
        mat_rows   = []
        for exp in exps:
            for mn in models:
                sub = df[(df['Experiment'] == exp) &
                         (df['Model'] == mn) &
                         (df['Task'] == task)]
                if not sub.empty:
                    # 使用简称
                    row_labels.append(
                        f"{_exp_short(exp)}\n{mn}")
                    mat_rows.append(sub['MAE'].values[0])
        if not mat_rows:
            continue

        mat  = np.array(mat_rows).reshape(-1, 1)
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
        im   = ax.imshow(mat, aspect='auto',
                         cmap='RdYlGn_r',
                         vmin=vmin, vmax=vmax)
        for ri, v in enumerate(mat_rows):
            fc = ('white' if v > (vmin+vmax)/2
                  else '#2c3e50')
            ax.text(0, ri, f"{v:.4f}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color=fc)
        ax.set_xticks([0])
        ax.set_xticklabels(
            [f"{TASK_LABELS[task]}\nMAE"], fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(TASK_LABELS[task], fontsize=10,
                     fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax,
                     fraction=0.1, pad=0.04)

    fig.suptitle(
        "Ablation Study — MAE Heatmap",
        fontsize=12, fontweight='bold',
        y=1.02, color='#1a1a2e')
    plt.tight_layout()
    out = os.path.join(
        save_dir, "Ablation_MAE_Heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  [Plot] Ablation heatmap saved.")


# ============================================================
# 8. 实验引擎
# ============================================================

def _collect_rows(results, exp_name, model_keys):
    rows = []
    for mn in model_keys:
        if mn not in results:
            continue
        bat_results = results[mn]
        if not bat_results:
            continue
        for task in TASK_NAMES:
            rows.append({
                'Experiment': exp_name,
                'Model':      mn,
                'Task':       task,
                'MAE':  np.mean([r[task]['mae']
                                 for r in bat_results]),
                'RMSE': np.mean([r[task]['rmse']
                                 for r in bat_results]),
            })
        hm = [r['tmax_high']['mae']
              for r in bat_results
              if r['tmax_high']['count'] > 0
              and not np.isnan(r['tmax_high']['mae'])]
        rows.append({
            'Experiment': exp_name,
            'Model':      mn,
            'Task':       'tmax_high',
            'MAE':  np.mean(hm) if hm else np.nan,
            'RMSE': np.nan,
        })
    return rows


def run_within_experiment(exp_name, pool_bats,
                          save_dir, device,
                          run_ablation=False):
    print(f"\n{'='*55}")
    print(f"  {exp_name}  Within-Condition K-Fold")
    print(f"  Ablation: {run_ablation}")
    print(f"{'='*55}")

    tw     = tuple(TASK_WEIGHTS[t] for t in TASK_NAMES)
    is_dyn = "Dynamic" in exp_name
    sig    = 2.5 if is_dyn else 1.0

    main_results = {m: [] for m in MAIN_MODEL_CLASSES}
    abl_results  = ({m: [] for m in ABLATION_MODEL_KEYS}
                    if run_ablation else {})

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for fold, (tri, tei) in enumerate(
            kf.split(pool_bats)):
        trb = [pool_bats[i] for i in tri]
        teb = [pool_bats[i] for i in tei]
        ds  = BatteryDataset(trb)
        ld  = DataLoader(ds, 128, shuffle=True,
                         drop_last=True)

        for mn, mc in MAIN_MODEL_CLASSES.items():
            ep = 100 if mn == 'Our' else 20
            m  = mc(ds.feat_dim).to(device)
            m  = train_engine(m, ld, device, ep,
                              weights=tw, htw=True)
            m.eval()
            for b in teb:
                r       = evaluate_battery(
                    m, b, ds, device, sig)
                r['id'] = b['id']
                main_results[mn].append(r)

        if run_ablation:
            for mn in ABLATION_MODEL_KEYS:
                if mn == 'Our':
                    for r in main_results['Our']:
                        if (r['id'] in
                                {b['id'] for b in teb}):
                            if not any(
                                    ar['id'] == r['id']
                                    for ar in
                                    abl_results['Our']):
                                abl_results['Our'].append(r)
                    continue
                mc  = ABLATION_MODEL_CLASSES[mn]
                htw = mn not in NO_HTW_MODELS
                m   = mc(ds.feat_dim).to(device)
                m   = train_engine(
                    m, ld, device, 100,
                    weights=tw, htw=htw)
                m.eval()
                for b in teb:
                    r       = evaluate_battery(
                        m, b, ds, device, sig)
                    r['id'] = b['id']
                    abl_results[mn].append(r)

        print(f"  Fold {fold+1}/4 done"
              f"{' (+ablation)' if run_ablation else ''}")

    plot_predictions(
        exp_name, main_results, save_dir,
        n_show=4, styles=MODEL_STYLES,
        is_cross=False)

    scores   = [{
        'id':    r['id'],
        'score': r['tmax']['mae'] + r['tmax']['rmse'],
    } for r in main_results['Our']]
    top4_ids = {x['id'] for x in
                sorted(scores,
                       key=lambda x: x['score'])[:4]}
    save_prediction_data(
        exp_name, main_results, save_dir,
        top_ids=top4_ids)
    save_test_metrics(
        exp_name, main_results, save_dir)

    main_rows = _collect_rows(
        main_results, exp_name, MAIN_MODEL_CLASSES)

    abl_rows = []
    if run_ablation and abl_results:
        plot_predictions(
            exp_name, abl_results, save_dir,
            n_show=4, styles=ABLATION_STYLES,
            tag='_Ablation', is_cross=False)
        abl_rows = _collect_rows(
            abl_results, exp_name, ABLATION_MODEL_KEYS)
        save_test_metrics(
            exp_name, abl_results, save_dir,
            suffix='_ablation')

    return main_rows, abl_rows, main_results, abl_results


def run_cross_experiment(exp_name, train_bats,
                         pool_bats, save_dir, device,
                         ud=True, htw=True, dw=0.3,
                         n_finetune=4,
                         mmd_weight=0.1):
    print(f"\n{'='*55}")
    print(f"  {exp_name}  Cross-Domain")
    print(f"{'='*55}")

    tw = tuple(TASK_WEIGHTS[t] for t in TASK_NAMES)

    random.seed(42)
    pool = copy.deepcopy(pool_bats)
    random.shuffle(pool)
    ft_bats = pool[:n_finetune]
    ev_bats = pool[n_finetune:]

    if len(ev_bats) == 0:
        ev_bats = ft_bats

    print(f"  Target: {len(ft_bats)} finetune, "
          f"{len(ev_bats)} eval")

    ds_src = BatteryDataset(train_bats)
    ld_src = DataLoader(ds_src, 128, shuffle=True,
                        drop_last=True)

    ds_ft = BatteryDataset(
        ft_bats, *ds_src.get_scalers())
    ld_ft = DataLoader(ds_ft, 32, shuffle=True,
                       drop_last=True)

    print("  [A] Main comparison (7 models, Transfer_FT)")
    main_results = {m: [] for m in MAIN_MODEL_CLASSES}

    for mn, mc in MAIN_MODEL_CLASSES.items():
        ep_pre = 80 if mn == 'Our' else 20
        m      = mc(ds_src.feat_dim,
                    use_domain=ud).to(device)
        m = train_engine(m, ld_src, device, ep_pre,
                         weights=tw, dw=dw,
                         htw=htw, ud=ud)
        m = train_engine(m, ld_ft, device, 40,
                         lr=2e-4, weights=tw,
                         dw=dw, htw=htw, ud=ud)
        m.eval()
        for b in ev_bats:
            r       = evaluate_battery(
                m, b, ds_src, device, 2.5, ud)
            r['id'] = b['id']
            main_results[mn].append(r)
        print(f"    {mn}: done")

    print("  [B] Transfer strategy comparison (Our only)")
    transfer_results = {}

    print("    [B1] No-Transfer")
    m_nt = OurModel(ds_src.feat_dim,
                    use_domain=False).to(device)
    m_nt = train_engine(m_nt, ld_src, device, 80,
                        weights=tw, dw=dw,
                        htw=htw, ud=False)
    m_nt.eval()
    res_nt = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_nt, b, ds_src, device, 2.5, False)
        r['id'] = b['id']
        res_nt.append(r)
    transfer_results['No-Transfer'] = res_nt

    print("    [B2] Target_Only")
    m_to = OurModel(ds_src.feat_dim,
                    use_domain=False).to(device)
    m_to = train_engine(m_to, ld_ft, device, 80,
                        weights=tw, dw=dw,
                        htw=htw, ud=False)
    m_to.eval()
    res_to = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_to, b, ds_src, device, 2.5, False)
        r['id'] = b['id']
        res_to.append(r)
    transfer_results['Target_Only'] = res_to

    print("    [B3] Domain_Embed")
    m_de = OurModel(ds_src.feat_dim,
                    use_domain=True).to(device)
    m_de = train_engine(m_de, ld_src, device, 80,
                        weights=tw, dw=dw,
                        htw=htw, ud=True)
    m_de.eval()
    res_de = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_de, b, ds_src, device, 2.5, True)
        r['id'] = b['id']
        res_de.append(r)
    transfer_results['Domain_Embed'] = res_de

    print("    [B4] Transfer_FT (reuse main Our)")
    transfer_results['Transfer_FT'] = main_results['Our']

    print("    [B5] Transfer_Align (MMD)")
    m_al = OurModel(ds_src.feat_dim,
                    use_domain=ud).to(device)
    m_al = train_engine(
        m_al, ld_src, device, 80,
        weights=tw, dw=dw, htw=htw, ud=ud,
        align_loader=ld_ft,
        mmd_weight=mmd_weight)
    m_al = train_engine(
        m_al, ld_ft, device, 40,
        lr=2e-4, weights=tw,
        dw=dw, htw=htw, ud=ud)
    m_al.eval()
    res_al = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_al, b, ds_src, device, 2.5, ud)
        r['id'] = b['id']
        res_al.append(r)
    transfer_results['Transfer_Align'] = res_al

    plot_predictions(
        exp_name, main_results, save_dir,
        n_show=4, styles=MODEL_STYLES,
        tag='_MainCompare', is_cross=True)

    plot_predictions(
        exp_name, transfer_results, save_dir,
        n_show=4, styles=TRANSFER_STYLES,
        tag='_TransferStrategy', is_cross=True)

    scores   = [{
        'id':    r['id'],
        'score': r['tmax']['mae'] + r['tmax']['rmse'],
    } for r in main_results['Our']]
    top_ids  = {x['id'] for x in
                sorted(scores,
                       key=lambda x: x['score'])[:8]}

    save_prediction_data(
        exp_name, main_results, save_dir,
        top_ids=top_ids)
    save_test_metrics(
        exp_name, main_results, save_dir)
    save_test_metrics(
        exp_name, transfer_results, save_dir,
        suffix='_TransferStrategy')

    main_rows = _collect_rows(
        main_results, exp_name, MAIN_MODEL_CLASSES)

    transfer_rows = _collect_rows(
        transfer_results, exp_name,
        TRANSFER_STRATEGY_KEYS)

    return main_rows, transfer_rows, main_results, \
           transfer_results


# ============================================================
# 9. 主函数
# ============================================================

def main():
    base_dir = "./extracted_features1/"
    save_dir = "./Results_Main_v4"
    os.makedirs(save_dir, exist_ok=True)

    pkls = sorted(glob.glob(
        os.path.join(base_dir, "*.pkl")))
    if not pkls:
        print(f"[ERROR] No pkl files in {base_dir}")
        return

    bf_pkls = [f for f in pkls
               if 'Batch-1' in f or 'Batch-2' in f]
    bd_pkls = [f for f in pkls
               if 'Batch-3' in f or 'Batch-4' in f]

    bf = extract_batteries(
        bf_pkls, "Fixed", domain_id=0)
    bd = extract_batteries(
        bd_pkls, "Dyn",   domain_id=1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    all_metrics          = []
    all_abl_metrics      = []
    all_transfer_metrics = []

    if bf:
        main_rows, abl_rows, _, _ = \
            run_within_experiment(
                "Exp1_Fixed", bf, save_dir, device,
                run_ablation=True)
        all_metrics.extend(main_rows)
        all_abl_metrics.extend(abl_rows)

    if bd:
        main_rows, abl_rows, _, _ = \
            run_within_experiment(
                "Exp2_Dynamic", bd, save_dir, device,
                run_ablation=True)
        all_metrics.extend(main_rows)
        all_abl_metrics.extend(abl_rows)

    if bf and bd:
        main_rows, transfer_rows, _, _ = \
            run_cross_experiment(
                "Exp3_Fixed2Dyn", bf, bd,
                save_dir, device,
                ud=True, mmd_weight=0.1)
        all_metrics.extend(main_rows)
        all_transfer_metrics.extend(transfer_rows)

    if bf and bd:
        main_rows, transfer_rows, _, _ = \
            run_cross_experiment(
                "Exp4_Dyn2Fixed", bd, bf,
                save_dir, device,
                ud=True, mmd_weight=0.1)
        all_metrics.extend(main_rows)
        all_transfer_metrics.extend(transfer_rows)

    if all_metrics:
        plot_overview_metrics(all_metrics, save_dir)

    if all_abl_metrics:
        plot_ablation_heatmap(
            all_abl_metrics, save_dir)

    if all_transfer_metrics:
        plot_transfer_strategy_heatmap(
            all_transfer_metrics, save_dir)

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        main_df = df[df['Task'].isin(TASK_NAMES)]
        pivot   = main_df.pivot_table(
            index=['Experiment', 'Model'],
            columns='Task',
            values=['MAE', 'RMSE'],
            aggfunc='mean').round(4)

        print(f"\n{'='*70}")
        print("  Main Results Summary")
        print('='*70)
        print(pivot.to_string())

        df.to_csv(
            os.path.join(save_dir, "Main_Summary.csv"),
            index=False)
        print(f"\n  [Save] → Main_Summary.csv")

    if all_abl_metrics:
        pd.DataFrame(all_abl_metrics).to_csv(
            os.path.join(save_dir,
                         "Ablation_Summary.csv"),
            index=False)
        print(f"  [Save] → Ablation_Summary.csv")

    if all_transfer_metrics:
        df_t = pd.DataFrame(all_transfer_metrics)
        df_t.to_csv(
            os.path.join(save_dir,
                         "Transfer_Strategy_Summary.csv"),
            index=False)
        print(f"  [Save] → Transfer_Strategy_Summary.csv")

        print(f"\n{'='*70}")
        print("  Transfer Strategy Results")
        print('='*70)
        pivot_t = df_t[
            df_t['Task'].isin(TASK_NAMES)
        ].pivot_table(
            index=['Experiment', 'Model'],
            columns='Task',
            values='MAE',
            aggfunc='mean').round(4)
        print(pivot_t.to_string())

    print(f"\n{'='*70}")
    print(f"  All done. Results → {save_dir}")
    print('='*70)


if __name__ == "__main__":
    main()