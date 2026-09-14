"""
NASA电池数据集验证脚本 v4
=========================================
基于 XJTU 主实验脚本 v4 对齐改动：

新增（对齐XJTU v4）：
1. 补全 CNN_LSTM / BiLSTM / PatchTST 三个基线模型
2. 消融实验（LOO组内）：OurModel六变体
3. 迁移策略对比（Transfer组内）：
   No-Transfer / Target_Only / Domain_Embed /
   Transfer_FT / Transfer_Align（MMD）
4. mmd_loss() 函数
5. train_engine() 支持 align_loader / mmd_weight
6. plot_transfer_strategy_heatmap() 迁移策略热力图
7. 报警层改用 battery_alert.py 的 BatteryAlertAnalyzer

保持不变：
- SEQ_LEN=20 / 数据加载 / BatteryDataset
- LOO 实验结构
- 原有 Transfer 跨组实验结构
- save_prediction_data / save_test_metrics
- plot_results_loo / plot_results_transfer
- plot_combined_heatmap
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

plt.rcParams['font.sans-serif'] = ['DejaVu Sans',
                                   'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid']         = True
plt.rcParams['grid.alpha']        = 0.25
plt.rcParams['grid.linestyle']    = '--'
plt.rcParams['axes.labelsize']    = 10
plt.rcParams['xtick.labelsize']   = 8
plt.rcParams['ytick.labelsize']   = 8

# ============================================================
# 全局配置
# ============================================================
# 开源版：请按需修改为 NASA 特征目录路径（即 NASA_dataprocessing.py 的输出目录）
NASA_FEAT_DIR = "./nasa_features"
SAVE_DIR      = "./Results_NASA"

BATTERY_GROUPS = {
    "Group_A": ["B0005", "B0006", "B0007", "B0018"],
    "Group_B": ["B0045", "B0046", "B0047", "B0048"],
    "Group_C": ["B0053", "B0054", "B0055", "B0056"],
}

TASK_NAMES   = ['tmax', 'soh', 'tmean']
N_OUTPUTS    = 3
SEQ_LEN      = 20
TASK_WEIGHTS = {'tmax': 1.0, 'soh': 1.0, 'tmean': 1.0}

HIGH_TEMP_THRESHOLDS = [35.0, 40.0]
HIGH_TEMP_WEIGHTS    = [1.5,  2.5]

TASK_LABELS = {
    'tmax':  'T$_{max}$ (°C)',
    'soh':   'SOH',
    'tmean': 'T$_{mean}$ (°C)',
}

# ── 主对比模型样式（对齐XJTU v4）──
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

# ── 消融样式 ──
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

# ── 迁移策略样式 ──
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
    'bg_even':  '#f8f9fa',
    'bg_odd':   '#ffffff',
}

TARGET_COLS  = ['Target_SOH', 'Target_T_max',
                'Target_T_mean', 'Target_T_rise']
EXCLUDE_COLS = ({'battery_id', 'cycle_count',
                 'cycle_num', 'cycle_index'} |
                set(TARGET_COLS))

# 消融模型键
ABLATION_MODEL_KEYS = [
    'Our_noSE', 'Our_TCNonly', 'Our_Mambaonly',
    'Our_noGate', 'Our_noHTW', 'Our'
]
NO_HTW_MODELS = {'Our_noHTW'}

# 迁移策略键
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
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# 1. 数据加载（不变）
# ============================================================

def load_battery_from_pkl(pkl_path, battery_name):
    df = pd.read_pickle(pkl_path).copy()
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE_COLS]
    df[feat_cols] = (df[feat_cols]
                     .replace([np.inf, -np.inf], np.nan)
                     .fillna(df[feat_cols].median())
                     .fillna(0.0))

    feat_arr  = df[feat_cols].values.astype(np.float32)
    tmax_arr  = gaussian_filter1d(
        df['Target_T_max'].values.astype(np.float32),  0.8)
    tmean_arr = gaussian_filter1d(
        df['Target_T_mean'].values.astype(np.float32), 0.8)
    soh_arr   = gaussian_filter1d(
        df['Target_SOH'].values.astype(np.float32),    0.5)
    delta_arr = tmax_arr - tmean_arr

    group = 'Unknown'
    for gname, bats in BATTERY_GROUPS.items():
        if battery_name in bats:
            group = gname
            break

    return {
        'id':        battery_name,
        'group':     group,
        'feat':      feat_arr,
        'tmax':      tmax_arr,
        'soh':       soh_arr,
        'tmean':     tmean_arr,
        'delta_t':   delta_arr,
        'domain_id': 0,
        'feat_cols': feat_cols,
        'n_cycles':  len(df),
    }


def load_nasa_batteries(feat_dir, target_batteries=None):
    pkl_files = sorted(glob.glob(
        os.path.join(feat_dir, "*_features.pkl")))
    if not pkl_files:
        raise FileNotFoundError(
            f"未在 {feat_dir} 找到 *_features.pkl")

    batteries = []
    for pkl_path in pkl_files:
        bat_name = os.path.basename(pkl_path).replace(
            '_features.pkl', '')
        if (target_batteries is not None and
                bat_name not in target_batteries):
            continue
        try:
            rec = load_battery_from_pkl(pkl_path, bat_name)
            if rec['n_cycles'] < SEQ_LEN + 5:
                print(f"  [SKIP] {bat_name}: "
                      f"仅{rec['n_cycles']}个循环")
                continue
            batteries.append(rec)
            print(f"  [OK] {bat_name:8s}: "
                  f"{rec['n_cycles']:4d} cycles, "
                  f"feat={rec['feat'].shape[1]}dim, "
                  f"group={rec['group']}")
        except Exception as e:
            print(f"  [ERR] {bat_name}: {e}")

    print(f"\n共加载 {len(batteries)} 个电池")
    return batteries


# ============================================================
# 2. 模型定义（新增 CNN_LSTM / BiLSTM / PatchTST
#              + 消融变体 + return_feat 支持）
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
                              dilation=dilation,
                              groups=ch)
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


def _forward_heads(heads, h, domain_emb=None,
                   domain_id=None):
    if domain_emb is not None and domain_id is not None:
        h = h + domain_emb(domain_id)
    return (heads['tmax'](h),  heads['soh'](h),
            heads['tmean'](h), heads['delta'](h))


# ── 主模型 ──

class OurModel(nn.Module):
    def __init__(self, input_dim, d_model=128,
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
        self.gate   = nn.Sequential(
            nn.Linear(d_model*2, d_model*2),
            nn.Sigmoid())
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
            self.heads, feat,
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
        _, h  = self.gru(x)
        feat  = self.dropout(h[-1])
        o     = self.fc(feat)
        z     = torch.zeros(o.shape[0], 1, device=x.device)
        out   = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32,
                 use_domain=False):
        super().__init__()
        self.emb = nn.Linear(input_dim, d_model)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4,
            batch_first=True, dropout=0.5)
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
        c = self.cnn(x.permute(0,2,1)).permute(0,2,1)
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
                 patch_len=4, stride=2,
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
        h = (self.patch_emb(patches) +
             self.pos_emb[:, :patches.shape[1], :])
        h    = self.encoder(h)
        h    = self.norm(h)
        feat = self.dropout(h.mean(dim=1))
        o    = self.fc(feat)
        z    = torch.zeros(o.shape[0], 1, device=x.device)
        out  = o[:,0:1], o[:,1:2], o[:,2:3], z
        if return_feat:
            return out, feat
        return out


# ── 消融变体 ──

class OurModel_noSE(nn.Module):
    def __init__(self, input_dim, d_model=64,
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
            d_model=d_model, d_state=64,
            d_conv=4, expand=2)
        self.gate   = nn.Sequential(
            nn.Linear(d_model*2, d_model*2),
            nn.Sigmoid())
        self.fusion = nn.Linear(d_model*2, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.heads  = _make_heads(d_model)

    def forward(self, x, domain_id=None,
                return_feat=False):
        h = self.input_proj(x)
        t = self.tcn_stream(
            h.permute(0,2,1)).permute(0,2,1)
        m = self.mamba_stream(h)
        c = torch.cat([t, m], -1)
        feat = self.norm(
            self.fusion(c * self.gate(c)))[:, -1, :]
        out  = _forward_heads(
            self.heads, feat,
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
            nn.Dropout(0.3))
        self.norm  = nn.LayerNorm(d_model)
        self.heads = _make_heads(d_model)

    def forward(self, x, domain_id=None,
                return_feat=False):
        h = self.se(self.input_proj(x))
        h = self.tcn_stream(
            h.permute(0,2,1)).permute(0,2,1)
        feat = self.norm(h)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class OurModel_Mambaonly(nn.Module):
    def __init__(self, input_dim, d_model=32,
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
            self.heads, feat,
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
        feat = self.norm(0.5 * t + 0.5 * m)[:, -1, :]
        out  = _forward_heads(
            self.heads, feat,
            self.domain_emb if self.use_domain else None,
            domain_id)
        if return_feat:
            return out, feat
        return out


class OurModel_noHTW(nn.Module):
    def __init__(self, input_dim, d_model=32,
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


# ============================================================
# 3. Dataset（不变）
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
                self.y.append([tn[j], sn[j],
                                mn[j], dn[j]])
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
# 4. 损失 + 训练（新增 mmd_loss + align_loader 支持）
# ============================================================

def get_temp_weights(traw, device):
    w = torch.ones_like(traw)
    for th, wt in zip(HIGH_TEMP_THRESHOLDS,
                      HIGH_TEMP_WEIGHTS):
        w = torch.where(traw >= th,
                        torch.tensor(wt, device=device),
                        w)
    return w


def compute_loss(pt, ps, pm, pd,
                 gt, gs, gm, gd,
                 traw, weights,
                 dw=0.3, htw=True, device='cpu'):
    def wh(pred, target, sw=None):
        base = nn.functional.huber_loss(
            pred.squeeze(), target.squeeze(),
            reduction='none')
        if sw is not None:
            base = base * sw
        return base.mean()
    tw = (get_temp_weights(traw, device) if htw
          else torch.ones(len(traw), device=device))
    return (weights[0] * wh(pt, gt, tw) +
            weights[1] * wh(ps, gs) +
            weights[2] * wh(pm, gm, tw) +
            dw * wh(pd, gd))


def mmd_loss(src_feat, tgt_feat):
    """
    多尺度 RBF 核 MMD 损失
    src_feat, tgt_feat: [B, D]
    """
    def rbf_kernel(x, y, bw):
        xx = (x ** 2).sum(1, keepdim=True)
        yy = (y ** 2).sum(1, keepdim=True)
        xy = x @ y.T
        dist = xx + yy.T - 2 * xy
        return torch.exp(-dist / (2 * bw ** 2))

    bandwidths = [0.5, 1.0, 2.0, 4.0]
    mmd = torch.tensor(0.0, device=src_feat.device)
    n_s = src_feat.shape[0]
    n_t = tgt_feat.shape[0]
    for bw in bandwidths:
        k_ss = rbf_kernel(src_feat, src_feat,
                          bw).sum() / (n_s * n_s)
        k_tt = rbf_kernel(tgt_feat, tgt_feat,
                          bw).sum() / (n_t * n_t)
        k_st = rbf_kernel(src_feat, tgt_feat,
                          bw).sum() / (n_s * n_t)
        mmd  = mmd + k_ss + k_tt - 2 * k_st
    return mmd / len(bandwidths)


def train_engine(model, loader, device,
                 epochs=60, lr=1e-3,
                 weights=(1., 1., 1.),
                 dw=0.3, htw=True, ud=False,
                 align_loader=None,
                 mmd_weight=0.1):
    """
    向后兼容：align_loader=None 时退化为普通训练
    ud=True 时传入 domain_id
    """
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
                    xb,
                    did if ud else None,
                    return_feat=True)

                tgt_did = torch.ones(
                    xt.shape[0], dtype=torch.long,
                    device=device)
                _, tgt_feat = model(
                    xt,
                    tgt_did if ud else None,
                    return_feat=True)

                task_loss = compute_loss(
                    pt, ps, pm, pd,
                    yb[:,0:1], yb[:,1:2],
                    yb[:,2:3], yb[:,3:4],
                    traw, weights, dw, htw, device)
                loss = (task_loss +
                        mmd_weight *
                        mmd_loss(src_feat, tgt_feat))
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
# 5. 评估（不变，但兼容 ud 参数）
# ============================================================

def evaluate_battery(model, bat, ds, device,
                     sigma=0.8, ud=False):
    sx, sy_t, sy_s, sy_m, sy_d = ds.get_scalers()
    tds = BatteryDataset(
        [bat], sx, sy_t, sy_s, sy_m, sy_d)
    ldr = DataLoader(tds, batch_size=256, shuffle=False)

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
        ts     = bat[task][L:]
        ps_inv = sy.inverse_transform(
            np.array(preds).reshape(-1,1)).flatten()
        ps_inv = gaussian_filter1d(ps_inv, sigma)
        n      = min(len(ts), len(ps_inv))
        ts, ps_inv = ts[:n], ps_inv[:n]
        result[task] = {
            'mae':  mean_absolute_error(ts, ps_inv),
            'rmse': np.sqrt(
                mean_squared_error(ts, ps_inv)),
            'ts':   ts,
            'ps':   ps_inv,
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
# 6. 保存函数（不变）
# ============================================================

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
                         test_bat_ids=None):
    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    if test_bat_ids is None:
        first_key    = next(iter(results))
        test_bat_ids = {r['id']
                        for r in results[first_key]}

    all_rows    = []
    metric_rows = []

    for mn, bat_results in results.items():
        model_rows = []
        for r in bat_results:
            bat_id = r['id']
            if bat_id not in test_bat_ids:
                continue
            N = len(r['tmax']['ts'])
            for i in range(N):
                row = {
                    'bat_id':       bat_id,
                    'model':        mn,
                    'exp':          exp_name,
                    'step':         i,
                    'tmax_true':    (r['tmax']['ts'][i]
                                     if i < len(r['tmax']['ts'])
                                     else np.nan),
                    'tmax_pred':    (r['tmax']['ps'][i]
                                     if i < len(r['tmax']['ps'])
                                     else np.nan),
                    'soh_true':     (r['soh']['ts'][i]
                                     if i < len(r['soh']['ts'])
                                     else np.nan),
                    'soh_pred':     (r['soh']['ps'][i]
                                     if i < len(r['soh']['ps'])
                                     else np.nan),
                    'tmean_true':   (r['tmean']['ts'][i]
                                     if i < len(r['tmean']['ts'])
                                     else np.nan),
                    'tmean_pred':   (r['tmean']['ps'][i]
                                     if i < len(r['tmean']['ps'])
                                     else np.nan),
                    'delta_t_true': (r['delta_t']['ts'][i]
                                     if i < len(r['delta_t']['ts'])
                                     else np.nan),
                    'delta_t_pred': (r['delta_t']['values'][i]
                                     if i < len(r['delta_t']['values'])
                                     else np.nan),
                }
                model_rows.append(row)
                all_rows.append(row)

            for task in TASK_NAMES:
                metric_rows.append({
                    'exp':      exp_name,
                    'model':    mn,
                    'bat_id':   bat_id,
                    'task':     task,
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
          f"{exp_name} ({len(test_bat_ids)} bats)")
    return pred_dir


# ============================================================
# 7. 可视化（对齐XJTU v4，复用 plot_predictions）
# ============================================================

def _make_metric_text(mae, rmse):
    return f"MAE={mae:.4f}\nRMSE={rmse:.4f}"


def _shade_error(ax, x, ts, ps, color, alpha=0.12):
    ax.fill_between(x, ts, ps,
                    where=(ps >= ts),
                    interpolate=True,
                    alpha=alpha, color=color)
    ax.fill_between(x, ts, ps,
                    where=(ps < ts),
                    interpolate=True,
                    alpha=alpha, color=color)


def plot_predictions(exp_name, results, save_dir,
                     n_show=4, styles=None,
                     tag='', is_cross=False):
    """统一预测曲线图（与XJTU v4完全一致）"""
    if styles is None:
        styles = MODEL_STYLES

    model_keys = [k for k in styles if k in results]
    if not model_keys:
        return

    ref_key = ('Transfer_Align'
                if 'Transfer_Align' in results
                else ('Our' if 'Our' in results
                      else model_keys[0]))

    scores = [{
        'id':    r['id'],
        'score': r['tmax']['mae'] + r['tmax']['rmse'],
    } for r in results[ref_key]]
    ids = [x['id'] for x in
           sorted(scores,
                  key=lambda x: x['score'])[:n_show]]

    n_tasks = len(TASK_NAMES)
    fig = plt.figure(
        figsize=(5*n_show+1, 3.5*n_tasks+2),
        facecolor='#fafafa')
    gs  = gridspec.GridSpec(
        n_tasks, n_show, figure=fig,
        left=0.06, right=0.97,
        top=0.91,  bottom=0.09,
        hspace=0.40, wspace=0.28)

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
            ax.set_facecolor(
                PALETTE['bg_even'] if ri % 2 == 0
                else PALETTE['bg_odd'])
            ts = ref_r[task]['ts']
            N  = len(ts)
            x  = np.arange(N)
            _shade_error(
                ax, x, ts, ref_r[task]['ps'],
                styles[ref_key]['color'], alpha=0.10)
            ax.plot(x, ts, color=PALETTE['measured'],
                    lw=1.6, alpha=0.60, zorder=10)
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
                        alpha=0.88, zorder=st['zorder'])
            ax.text(0.98, 0.97,
                    _make_metric_text(
                        ref_r[task]['mae'],
                        ref_r[task]['rmse']),
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va='top', ha='right',
                    color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', ec='#cccccc',
                              alpha=0.85))
            if ri == 0:
                ax.set_title(bid, fontsize=9,
                             fontweight='bold', pad=4,
                             color='#2c3e50')
            if ci == 0:
                ax.set_ylabel(TASK_LABELS[task],
                              fontsize=9, labelpad=4)
            if ri == n_tasks - 1:
                ax.set_xlabel('Cycle', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.yaxis.set_major_locator(
                plt.MaxNLocator(nbins=4, prune='both'))
            ax.xaxis.set_major_locator(
                plt.MaxNLocator(nbins=5, integer=True))

    domain_tag = ("Cross-Group" if is_cross
                  else "Within-Group")
    fig.suptitle(
        f"[NASA] {exp_name}{tag}  —  {domain_tag}  —  "
        f"Best {n_show} Batteries",
        fontsize=12, fontweight='bold',
        color='#1a1a2e', y=0.97)
    fig.legend(
        handles=legend_handles, loc='lower center',
        ncol=min(len(legend_handles), 8),
        fontsize=7.5, frameon=True, framealpha=0.9,
        edgecolor='#cccccc',
        bbox_to_anchor=(0.50, 0.01))

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir,
                       f"{exp_name}{tag}_pred.png")
    plt.savefig(out, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Plot] {os.path.basename(out)}")


# 保留原有 LOO / Transfer 特定布局（向后兼容）
def plot_results_loo(exp_name, results, save_dir):
    plot_predictions(exp_name, results, save_dir,
                     n_show=4, styles=MODEL_STYLES,
                     is_cross=False)


def plot_results_transfer(exp_name, results, save_dir):
    plot_predictions(exp_name, results, save_dir,
                     n_show=4, styles=MODEL_STYLES,
                     is_cross=True)


# ============================================================
# 8. 热力图
# ============================================================

def plot_combined_heatmap(all_metric_rows,
                          exp_list,
                          title_tag,
                          save_path):
    """LOO / Transfer 组合热力图（不变）"""
    df = pd.DataFrame(all_metric_rows)
    df = df[df['Experiment'].isin(exp_list)]
    if df.empty:
        print(f"  [Heatmap] 无数据，跳过 {title_tag}")
        return

    models = list(MAIN_MODEL_CLASSES.keys())
    tasks  = TASK_NAMES

    summary = []
    for exp in exp_list:
        for mn in models:
            sub = df[(df['Experiment'] == exp) &
                     (df['Model'] == mn)]
            if sub.empty:
                continue
            entry = {'Experiment': exp, 'Model': mn}
            for t in tasks:
                col = f'MAE_{t}'
                entry[f'MAE_{t}'] = (sub[col].mean()
                                     if col in sub.columns
                                     else np.nan)
            summary.append(entry)

    df_s = pd.DataFrame(summary)
    row_labels = [f"{r['Experiment']}\n{r['Model']}"
                  for _, r in df_s.iterrows()]
    mat  = df_s[[f'MAE_{t}' for t in tasks]].values

    fig_h = max(0.5 * len(row_labels) + 1.5, 4)
    fig_w = max(3.5 * len(tasks) + 1.5, 6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h),
                           facecolor='white')
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    im   = ax.imshow(mat, aspect='auto',
                     cmap='RdYlGn_r',
                     vmin=vmin, vmax=vmax)
    for ri in range(len(row_labels)):
        for ci in range(len(tasks)):
            v = mat[ri, ci]
            if not np.isnan(v):
                fc = ('white'
                      if v > (vmin+vmax)/2
                      else '#2c3e50')
                ax.text(ci, ri, f"{v:.4f}",
                        ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color=fc)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(
        [TASK_LABELS[t] for t in tasks], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title(f"MAE Heatmap — {title_tag}",
                 fontsize=11, fontweight='bold',
                 color='#2c3e50', pad=10)
    plt.colorbar(im, ax=ax, fraction=0.03,
                 pad=0.04, label='MAE')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Heatmap] → {os.path.basename(save_path)}")


def plot_transfer_strategy_heatmap(
        transfer_metrics, save_dir):
    """迁移策略热力图（对齐XJTU v4）"""
    df = pd.DataFrame(transfer_metrics)
    if df.empty:
        return

    for exp_name in df['Experiment'].unique():
        sub = df[df['Experiment'] == exp_name]
        mat = np.full(
            (len(TRANSFER_STRATEGY_KEYS),
             len(TASK_NAMES)), np.nan)
        for si, strat in enumerate(
                TRANSFER_STRATEGY_KEYS):
            for ti, task in enumerate(TASK_NAMES):
                row = sub[(sub['Model'] == strat) &
                          (sub['Task'] == task)]
                if not row.empty:
                    mat[si, ti] = row['MAE'].values[0]

        fig, ax = plt.subplots(
            figsize=(5,
                     0.7*len(TRANSFER_STRATEGY_KEYS)+1.5),
            facecolor='#fafafa')
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
        im   = ax.imshow(mat, aspect='auto',
                         cmap='RdYlGn_r',
                         vmin=vmin, vmax=vmax)
        for si in range(len(TRANSFER_STRATEGY_KEYS)):
            for ti in range(len(TASK_NAMES)):
                v = mat[si, ti]
                if not np.isnan(v):
                    fc = ('white'
                          if v > (vmin+vmax)/2
                          else '#2c3e50')
                    ax.text(ti, si, f"{v:.4f}",
                            ha='center', va='center',
                            fontsize=9,
                            fontweight='bold', color=fc)
        ax.set_xticks(range(len(TASK_NAMES)))
        ax.set_xticklabels(
            [TASK_LABELS[t] for t in TASK_NAMES],
            fontsize=9)
        ax.set_yticks(range(len(TRANSFER_STRATEGY_KEYS)))
        ax.set_yticklabels(
            [s.replace('_', '\n')
             for s in TRANSFER_STRATEGY_KEYS],
            fontsize=8)
        ax.set_title(
            f"[NASA] {exp_name} — "
            f"Transfer Strategy MAE",
            fontsize=11, fontweight='bold',
            color='#1a1a2e', pad=8)
        plt.colorbar(im, ax=ax,
                     fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = os.path.join(
            save_dir,
            f"{exp_name}_Transfer_Heatmap.png")
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [Plot] {os.path.basename(out)}")


def plot_ablation_heatmap(abl_metrics, save_dir):
    """消融热力图（对齐XJTU v4）"""
    df = pd.DataFrame(abl_metrics)
    if df.empty:
        return

    exps   = df['Experiment'].unique()
    models = ABLATION_MODEL_KEYS
    tasks  = TASK_NAMES

    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(4.5*len(tasks),
                 1.2*len(exps)*len(models)/2+2),
        facecolor='#fafafa')
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
                    row_labels.append(
                        f"{exp}\n{mn}")
                    mat_rows.append(
                        sub['MAE'].values[0])
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
            [f"{TASK_LABELS[task]}\nMAE"],
            fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(TASK_LABELS[task], fontsize=10,
                     fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax,
                     fraction=0.1, pad=0.04)

    fig.suptitle(
        "[NASA] Ablation Study — MAE Heatmap",
        fontsize=12, fontweight='bold',
        y=1.02, color='#1a1a2e')
    plt.tight_layout()
    out = os.path.join(save_dir,
                       "NASA_Ablation_Heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Plot] {os.path.basename(out)}")


# ============================================================
# 9. 指标收集辅助
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


# ============================================================
# 10. 实验引擎
# ============================================================

def run_loo(batteries, save_dir, device,
            exp_prefix="LOO",
            run_ablation=False):
    """
    LOO 组内实验
    新增：run_ablation=True 时跑消融变体
    """
    print(f"\n{'='*55}")
    print(f"  {exp_prefix}: Leave-One-Out")
    print(f"  Ablation: {run_ablation}")
    print(f"{'='*55}")

    tw           = tuple(TASK_WEIGHTS[t]
                         for t in TASK_NAMES)
    main_results = {m: [] for m in MAIN_MODEL_CLASSES}
    abl_results  = ({m: [] for m in ABLATION_MODEL_KEYS}
                    if run_ablation else {})
    all_test_ids = {b['id'] for b in batteries}
    all_rows     = []

    for test_i, test_bat in enumerate(batteries):
        train_bats = [b for j, b in enumerate(batteries)
                      if j != test_i]
        if not train_bats:
            continue

        print(f"\n  Fold {test_i+1}/{len(batteries)}: "
              f"Test={test_bat['id']}")

        ds = BatteryDataset(train_bats)
        ld = DataLoader(ds, batch_size=64,
                        shuffle=True, drop_last=True)

        # ── 主对比 ──
        for mn, mc in MAIN_MODEL_CLASSES.items():
            ep = 150 if mn == 'Our' else 30
            m  = mc(ds.feat_dim).to(device)
            m  = train_engine(m, ld, device,
                              epochs=ep, weights=tw)
            m.eval()
            r       = evaluate_battery(
                m, test_bat, ds, device, 0.8)
            r['id'] = test_bat['id']
            main_results[mn].append(r)

            row = {
                'Experiment': exp_prefix,
                'Battery':    test_bat['id'],
                'Group':      test_bat.get('group', 'N/A'),
                'Model':      mn,
            }
            for task in TASK_NAMES:
                row[f'MAE_{task}']  = r[task]['mae']
                row[f'RMSE_{task}'] = r[task]['rmse']
            all_rows.append(row)
            print(f"    {mn:12s}: "
                  f"Tmax={r['tmax']['mae']:.4f}  "
                  f"SOH={r['soh']['mae']:.5f}  "
                  f"Tmean={r['tmean']['mae']:.4f}")

        # ── 消融 ──
        if run_ablation:
            for mn in ABLATION_MODEL_KEYS:
                if mn == 'Our':
                    r = next(
                        (x for x in main_results['Our']
                         if x['id'] == test_bat['id']),
                        None)
                    if (r and not any(
                            ar['id'] == r['id']
                            for ar in
                            abl_results['Our'])):
                        abl_results['Our'].append(r)
                    continue
                mc  = ABLATION_MODEL_CLASSES[mn]
                htw = mn not in NO_HTW_MODELS
                m   = mc(ds.feat_dim).to(device)
                m   = train_engine(
                    m, ld, device, epochs=150,
                    weights=tw, htw=htw)
                m.eval()
                r       = evaluate_battery(
                    m, test_bat, ds, device, 0.8)
                r['id'] = test_bat['id']
                abl_results[mn].append(r)

        print(f"  Fold {test_i+1} done"
              f"{' (+ablation)' if run_ablation else ''}")

    # ── 可视化 ──
    if main_results['Our']:
        plot_predictions(
            exp_prefix, main_results, save_dir,
            n_show=4, styles=MODEL_STYLES,
            is_cross=False)
        save_prediction_data(
            exp_prefix, main_results, save_dir,
            test_bat_ids=all_test_ids)
        save_test_metrics(
            exp_prefix, main_results, save_dir)

    if run_ablation and abl_results:
        plot_predictions(
            exp_prefix, abl_results, save_dir,
            n_show=4, styles=ABLATION_STYLES,
            tag='_Ablation', is_cross=False)
        save_test_metrics(
            exp_prefix, abl_results, save_dir,
            suffix='_ablation')

    main_metric_rows = _collect_rows(
        main_results, exp_prefix, MAIN_MODEL_CLASSES)
    abl_metric_rows  = (_collect_rows(
        abl_results, exp_prefix, ABLATION_MODEL_KEYS)
                        if run_ablation else [])

    return (main_results, abl_results,
            all_rows, main_metric_rows, abl_metric_rows)


def run_group_transfer(src_bats, tgt_bats,
                       save_dir, device,
                       exp_name="Transfer",
                       n_ft=1, ud=True,
                       mmd_weight=0.1):
    """
    跨组迁移实验（对齐XJTU v4）
    A. 主对比：7个模型 × Transfer_FT
    B. 迁移策略对比（只跑 Our）：5路策略
    """
    print(f"\n{'='*55}")
    print(f"  {exp_name}  Cross-Group")
    print(f"{'='*55}")

    if len(tgt_bats) <= n_ft:
        print("  [WARN] 目标电池数量不足，跳过")
        return None, None, None, None

    tw = tuple(TASK_WEIGHTS[t] for t in TASK_NAMES)

    random.seed(42)
    tgt_shuffled = copy.deepcopy(tgt_bats)
    random.shuffle(tgt_shuffled)
    ft_bats = tgt_shuffled[:n_ft]
    ev_bats = tgt_shuffled[n_ft:]
    ev_ids  = {b['id'] for b in ev_bats}

    print(f"  Finetune: {[b['id'] for b in ft_bats]}")
    print(f"  Eval:     {[b['id'] for b in ev_bats]}")

    ds_src = BatteryDataset(src_bats)
    ld_src = DataLoader(ds_src, batch_size=64,
                        shuffle=True, drop_last=True)
    ds_ft  = BatteryDataset(ft_bats,
                            *ds_src.get_scalers())
    ld_ft  = DataLoader(ds_ft, batch_size=16,
                        shuffle=True, drop_last=True)

    # ════════════════════════════════════════
    # A. 主对比：7模型 × Transfer_FT
    # ════════════════════════════════════════
    print("  [A] Main comparison (7 models)")
    main_results = {m: [] for m in MAIN_MODEL_CLASSES}
    all_rows     = []

    for mn, mc in MAIN_MODEL_CLASSES.items():
        ep = 150 if mn == 'Our' else 30
        m  = mc(ds_src.feat_dim,
                use_domain=ud).to(device)
        m  = train_engine(m, ld_src, device,
                          epochs=ep, weights=tw,
                          ud=ud)
        m  = train_engine(m, ld_ft, device,
                          epochs=50, lr=5e-4,
                          weights=tw, ud=ud)
        m.eval()
        for b in ev_bats:
            r       = evaluate_battery(
                m, b, ds_src, device, 0.8, ud)
            r['id'] = b['id']
            main_results[mn].append(r)

            row = {
                'Experiment': exp_name,
                'Battery':    b['id'],
                'Group':      b.get('group', 'N/A'),
                'Model':      mn,
            }
            for task in TASK_NAMES:
                row[f'MAE_{task}']  = r[task]['mae']
                row[f'RMSE_{task}'] = r[task]['rmse']
            all_rows.append(row)
            print(f"  {mn:12s} → {b['id']}: "
                  f"Tmax={r['tmax']['mae']:.4f}  "
                  f"SOH={r['soh']['mae']:.5f}")

    # ════════════════════════════════════════
    # B. 迁移策略对比（只跑 Our）
    # ════════════════════════════════════════
    print("  [B] Transfer strategy (Our only)")
    transfer_results = {}

    # B1. No-Transfer
    print("    [B1] No-Transfer")
    m_nt = OurModel(ds_src.feat_dim,
                    use_domain=False).to(device)
    m_nt = train_engine(m_nt, ld_src, device,
                        epochs=150, weights=tw,
                        ud=False)
    m_nt.eval()
    res_nt = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_nt, b, ds_src, device, 0.8, False)
        r['id'] = b['id']
        res_nt.append(r)
    transfer_results['No-Transfer'] = res_nt

    # B2. Target_Only
    print("    [B2] Target_Only")
    m_to = OurModel(ds_src.feat_dim,
                    use_domain=False).to(device)
    m_to = train_engine(m_to, ld_ft, device,
                        epochs=80, weights=tw,
                        ud=False)
    m_to.eval()
    res_to = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_to, b, ds_src, device, 0.8, False)
        r['id'] = b['id']
        res_to.append(r)
    transfer_results['Target_Only'] = res_to

    # B3. Domain_Embed（不微调）
    print("    [B3] Domain_Embed")
    m_de = OurModel(ds_src.feat_dim,
                    use_domain=True).to(device)
    m_de = train_engine(m_de, ld_src, device,
                        epochs=150, weights=tw,
                        ud=True)
    m_de.eval()
    res_de = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_de, b, ds_src, device, 0.8, True)
        r['id'] = b['id']
        res_de.append(r)
    transfer_results['Domain_Embed'] = res_de

    # B4. Transfer_FT（复用主对比 Our 结果）
    print("    [B4] Transfer_FT (reuse)")
    transfer_results['Transfer_FT'] = \
        main_results['Our']

    # B5. Transfer_Align（MMD + 微调）
    print("    [B5] Transfer_Align (MMD)")
    m_al = OurModel(ds_src.feat_dim,
                    use_domain=ud).to(device)
    m_al = train_engine(
        m_al, ld_src, device, epochs=150,
        weights=tw, ud=ud,
        align_loader=ld_ft,
        mmd_weight=mmd_weight)
    m_al = train_engine(
        m_al, ld_ft, device, epochs=50,
        lr=5e-4, weights=tw, ud=ud)
    m_al.eval()
    res_al = []
    for b in ev_bats:
        r       = evaluate_battery(
            m_al, b, ds_src, device, 0.8, ud)
        r['id'] = b['id']
        res_al.append(r)
    transfer_results['Transfer_Align'] = res_al

    # ── 可视化 ──
    plot_predictions(
        exp_name, main_results, save_dir,
        n_show=4, styles=MODEL_STYLES,
        tag='_MainCompare', is_cross=True)
    plot_predictions(
        exp_name, transfer_results, save_dir,
        n_show=4, styles=TRANSFER_STYLES,
        tag='_TransferStrategy', is_cross=True)

    # ── 保存 ──
    save_prediction_data(
        exp_name, main_results, save_dir,
        test_bat_ids=ev_ids)
    save_test_metrics(
        exp_name, main_results, save_dir)
    save_test_metrics(
        exp_name, transfer_results, save_dir,
        suffix='_TransferStrategy')

    # ── 汇总指标 ──
    main_metric_rows     = _collect_rows(
        main_results, exp_name, MAIN_MODEL_CLASSES)
    transfer_metric_rows = _collect_rows(
        transfer_results, exp_name,
        TRANSFER_STRATEGY_KEYS)

    return (main_results, transfer_results,
            all_rows, main_metric_rows,
            transfer_metric_rows)


# ============================================================
# 11. 主函数
# ============================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NASA Battery Dataset Validation v4")
    print(f"  Feature dir : {NASA_FEAT_DIR}")
    print(f"  Save dir    : {SAVE_DIR}")
    print(f"{'='*60}\n")

    all_batteries = load_nasa_batteries(NASA_FEAT_DIR)
    if not all_batteries:
        raise RuntimeError("未加载到任何电池数据")

    group_bats = {}
    for b in all_batteries:
        g = b.get('group', 'Unknown')
        group_bats.setdefault(g, []).append(b)

    print(f"\n电池分组:")
    for gname, bats in group_bats.items():
        print(f"  {gname}: {[b['id'] for b in bats]}")

    all_rows              = []
    all_abl_metric_rows   = []
    all_transfer_metrics  = []
    loo_results_all       = {}
    all_metric_rows_loo   = []
    all_metric_rows_tf    = []
    loo_exp_names         = []
    transfer_exp_names    = []

    # ══════════════════════════════════════════════════
    # LOO 组内（含消融）
    # ══════════════════════════════════════════════════
    for gname, bats in group_bats.items():
        if len(bats) < 2:
            print(f"\n  [{gname}] 电池数<2，跳过LOO")
            continue
        exp_name = f"LOO_{gname}"
        loo_exp_names.append(exp_name)

        (res, abl_res, rows,
         main_metric_rows,
         abl_metric_rows) = run_loo(
            bats, SAVE_DIR, device, exp_name,
            run_ablation=True)

        all_rows.extend(rows)
        loo_results_all[gname] = res
        all_metric_rows_loo.extend(rows)
        all_abl_metric_rows.extend(abl_metric_rows)

        print(f"\n  [{exp_name}] Summary:")
        for mn in MAIN_MODEL_CLASSES:
            if not res[mn]:
                continue
            for task in TASK_NAMES:
                mae = np.mean([r[task]['mae']
                               for r in res[mn]])
                print(f"    {mn:12s} {task}: "
                      f"MAE={mae:.4f}")

    # ══════════════════════════════════════════════════
    # Transfer 跨组（含迁移策略对比）
    # ══════════════════════════════════════════════════
    transfer_pairs = [
        ('Group_A', 'Group_B', 'Transfer_A2B'),
        ('Group_B', 'Group_C', 'Transfer_B2C'),
        ('Group_A', 'Group_C', 'Transfer_A2C'),
    ]
    for src_g, tgt_g, exp_name in transfer_pairs:
        if (src_g not in group_bats or
                tgt_g not in group_bats):
            continue
        transfer_exp_names.append(exp_name)

        ret = run_group_transfer(
            group_bats[src_g],
            group_bats[tgt_g],
            SAVE_DIR, device,
            exp_name=exp_name,
            n_ft=1, ud=True,
            mmd_weight=0.1)

        if ret[0] is None:
            continue

        (_, _, tf_rows,
         main_metric_rows,
         transfer_metric_rows) = ret

        if tf_rows:
            all_rows.extend(tf_rows)
            all_metric_rows_tf.extend(tf_rows)
        if transfer_metric_rows:
            all_transfer_metrics.extend(
                transfer_metric_rows)

    # ══════════════════════════════════════════════════
    # 热力图
    # ══════════════════════════════════════════════════
    if all_metric_rows_loo:
        plot_combined_heatmap(
            all_metric_rows_loo,
            exp_list=loo_exp_names,
            title_tag="LOO Within-Group",
            save_path=os.path.join(
                SAVE_DIR,
                "Heatmap_LOO_Combined.png"))

    if all_metric_rows_tf:
        plot_combined_heatmap(
            all_metric_rows_tf,
            exp_list=transfer_exp_names,
            title_tag="Transfer Cross-Group",
            save_path=os.path.join(
                SAVE_DIR,
                "Heatmap_Transfer_Combined.png"))

    if all_abl_metric_rows:
        plot_ablation_heatmap(
            all_abl_metric_rows, SAVE_DIR)

    if all_transfer_metrics:
        plot_transfer_strategy_heatmap(
            all_transfer_metrics, SAVE_DIR)

    # ══════════════════════════════════════════════════
    # 报警分析（调用 battery_alert.py）
    # ══════════════════════════════════════════════════
    try:
        from battery_alert import (
            BatteryAlertAnalyzer,
            NASAGroupAConfig, NASAGroupBConfig,
            NASAGroupCConfig, NASA_BAT_CFG_MAP,
            deduplicate_infos)

        print(f"\n{'='*55}")
        print("  Alert Analysis (via battery_alert.py)")
        print(f"{'='*55}")

        analyzer = BatteryAlertAnalyzer(
            save_dir=os.path.join(SAVE_DIR, "Alert"),
            cfg=NASAGroupAConfig())

        pred_dir    = os.path.join(
            SAVE_DIR, "predictions")
        all_infos   = []

        # LOO预警
        for exp_name in loo_exp_names:
            csv_path = os.path.join(
                pred_dir,
                f"{exp_name}_Our_predictions.csv")
            if not os.path.exists(csv_path):
                print(f"  [SKIP] {csv_path}")
                continue
            infos = analyzer.run_from_csv(
                csv_path,
                dataset_tag=f"NASA_{exp_name}",
                model_filter="Our",
                max_plot=4,
                bat_cfg_map=NASA_BAT_CFG_MAP)
            infos = deduplicate_infos(
                infos, strategy='best')
            analyzer.plot_summary(
                infos,
                dataset_tag=f"NASA_{exp_name}")
            analyzer.print_summary(
                infos,
                dataset_tag=f"NASA_{exp_name}")
            all_infos.extend(infos)

        # Transfer预警
        for exp_name in transfer_exp_names:
            csv_path = os.path.join(
                pred_dir,
                f"{exp_name}_Our_predictions.csv")
            if not os.path.exists(csv_path):
                print(f"  [SKIP] {csv_path}")
                continue
            infos = analyzer.run_from_csv(
                csv_path,
                dataset_tag=f"NASA_{exp_name}",
                model_filter="Our",
                max_plot=4,
                bat_cfg_map=NASA_BAT_CFG_MAP)
            infos = deduplicate_infos(
                infos, strategy='best')
            analyzer.plot_summary(
                infos,
                dataset_tag=f"NASA_{exp_name}")
            analyzer.print_summary(
                infos,
                dataset_tag=f"NASA_{exp_name}")
            all_infos.extend(infos)

        # 全局汇总
        all_infos = deduplicate_infos(
            all_infos, strategy='best')
        if all_infos:
            analyzer.plot_summary(
                all_infos, dataset_tag="NASA_All")
            analyzer.print_summary(
                all_infos, dataset_tag="NASA_All")

    except ImportError:
        print("  [WARN] battery_alert.py 未找到，"
              "跳过报警分析")

    # ══════════════════════════════════════════════════
    # 汇总 CSV
    # ══════════════════════════════════════════════════
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(
            os.path.join(SAVE_DIR, "NASA_Summary.csv"),
            index=False)

        print(f"\n{'='*70}")
        print("  NASA 验证总结")
        print('='*70)
        for exp in df['Experiment'].unique():
            sub_exp = df[df['Experiment'] == exp]
            print(f"\n  [{exp}]")
            for mn in MAIN_MODEL_CLASSES:
                sub   = sub_exp[sub_exp['Model'] == mn]
                if sub.empty:
                    continue
                parts = [
                    f"{t}="
                    f"{sub[f'MAE_{t}'].mean():.4f}"
                    for t in TASK_NAMES
                    if f'MAE_{t}' in sub.columns]
                print(f"    {mn:12s}: "
                      f"{', '.join(parts)}")

        print(f"\n  Our vs Best Baseline:")
        for task in TASK_NAMES:
            col = f'MAE_{task}'
            if col not in df.columns:
                continue
            our_mae = df[
                df['Model'] == 'Our'][col].mean()
            bl_maes = (df[df['Model'] != 'Our']
                       .groupby('Model')[col].mean())
            if bl_maes.empty:
                continue
            best_bl = bl_maes.min()
            best_mn = bl_maes.idxmin()
            imp = ((best_bl - our_mae) /
                   best_bl * 100)
            print(f"    {task}: Our={our_mae:.4f}, "
                  f"Best={best_bl:.4f}({best_mn}), "
                  f"Δ={imp:+.1f}%")

    if all_transfer_metrics:
        df_t = pd.DataFrame(all_transfer_metrics)
        df_t.to_csv(
            os.path.join(
                SAVE_DIR,
                "Transfer_Strategy_Summary.csv"),
            index=False)
        print(f"\n  [Save] → "
              f"Transfer_Strategy_Summary.csv")
        pivot = df_t[
            df_t['Task'].isin(TASK_NAMES)
        ].pivot_table(
            index=['Experiment', 'Model'],
            columns='Task',
            values='MAE',
            aggfunc='mean').round(4)
        print(pivot.to_string())

    if all_abl_metric_rows:
        pd.DataFrame(all_abl_metric_rows).to_csv(
            os.path.join(
                SAVE_DIR,
                "NASA_Ablation_Summary.csv"),
            index=False)
        print(f"  [Save] → NASA_Ablation_Summary.csv")

    print(f"\n{'='*70}")
    print(f"  All done. Results → {SAVE_DIR}")
    print('='*70)


if __name__ == "__main__":
    main()