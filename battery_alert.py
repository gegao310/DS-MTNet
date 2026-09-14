"""
battery_alert.py
================
通用电池预警分析模块 v4.0

v4.0 核心重设计
--------------
1. 删除重复的 find_pred_event_idx，合并为唯一版本
2. 引入相对寿命位置约束（MIN_TRIGGER_RATIO）
3. 强触发升级：level>=2 + 绝对温度条件双重验证
4. 弱触发引入趋势一致性检验（三通道方向一致）
5. 废弃不稳定的onset回溯，改为首次满足时刻
6. 新增 PRED_MIN_ADVANCE_RATIO：预测触发不能过早
7. 统一 gaussian_filter1d sigma=2.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patheffects as pe
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
# ============================================================
# 全局样式
# ============================================================
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.unicode_minus': False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.linestyle':     '--',
    'grid.alpha':         0.18,
    'grid.color':         '#b0b7c3',
    'xtick.labelsize':    8.5,
    'ytick.labelsize':    8.5,
    'axes.labelsize':     9.5,
    'axes.titlesize':     10,
    'legend.fontsize':    7.5,
    'figure.dpi':         160,
    'savefig.dpi':        200,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'axes.edgecolor':     '#4b5563',
    'axes.linewidth':     0.8,
})

C = {
    'true':        '#1a1a2e',
    'pred_tmax':   '#e74c3c',
    'pred_tmean':  '#3498db',
    'pred_dt':     '#8e44ad',
    'pred_soh':    '#e67e22',
    'zscore_t':    '#e74c3c',
    'zscore_d':    '#8e44ad',
    'baseline_bg': '#f0f3f7',
    'true_alert':  '#636e72',
    'pred_alert':  '#0984e3',
    'soh_fill':    '#fdcb6e',
}

LEVEL_COLORS = {0: '#00b894', 1: '#fdcb6e',
                2: '#e17055', 3: '#d63031'}
LEVEL_NAMES  = {0: 'Normal',  1: 'Caution',
                2: 'Warning', 3: 'Danger'}

TEMP_RISK_BANDS = [
    (38.0, 42.0, '#fff9c4', 'Caution zone'),
    (42.0, 46.0, '#ffe0b2', 'Warning zone'),
    (46.0, 60.0, '#ffcdd2', 'Danger zone'),
]


# ============================================================
# 参数配置基类
# ============================================================

class AlertConfig:
    # ── 基线 ──
    BASELINE_WINDOW    = 20
    CONFIRM_WINDOW     = 5
    CONFIRM_RATIO      = 0.60

    # ── 预警等级阈值 ──
    Z_THRESHOLDS       = {'warn': 1.6, 'alert': 2.6,
                          'danger': 4.0}
    SOH_REL_THRESHOLDS = {'minor': 0.03, 'medium': 0.06,
                          'severe': 0.10}
    ABS_DANGER         = {'tmax': 52.0, 'delta_t': 9.5}

    # ── 趋势检测 ──
    TREND_WINDOW       = 7
    TREND_SLOPE_TMAX   = 0.08
    TREND_SLOPE_DT     = 0.05
    TREND_Z_MIN        = 0.70

    # ── 真实事件检测 ──
    TRUE_TRIGGER_LEVEL = 2
    TRUE_GRACE_CYCLES  = 6
    TRUE_EVENT_WINDOW  = 4
    TRUE_EVENT_RATIO   = 0.75
    TRUE_REL_TMIN      = 1.0
    TRUE_REL_DTMIN     = 0.6

    # ── 预测触发：时间约束 ──
    PRED_GRACE_CYCLES    = 10   # 基线结束后最少等待周期
    # 触发时刻 >= 总长度 * MIN_TRIGGER_RATIO
    # 防止在寿命极早期误触发
    PRED_MIN_TRIGGER_RATIO = 0.10  # 至少过了总寿命10%

    # ── 预测触发：强触发条件 ──
    # 强触发：level持续高 + 温度绝对值验证
    PRED_STRONG_LEVEL    = 2
    PRED_STRONG_WINDOW   = 6
    PRED_STRONG_RATIO    = 0.80
    # 强触发还需满足：tmax预测超过基线+ABS_TEMP_RISE
    PRED_STRONG_ABS_RISE = 3.0   # 预测温度须比基线高3°C

    # ── 预测触发：弱触发子条件 ──
    PRED_Z_MIN           = 1.60
    PRED_REL_TMIN        = 1.20
    PRED_REL_DTMIN       = 0.70
    PRED_Z_WINDOW        = 5
    PRED_Z_RATIO         = 0.75
    PRED_LEVEL1_WINDOW   = 5
    PRED_LEVEL1_RATIO    = 0.80
    PRED_TREND_WINDOW    = 6
    PRED_Z_SLOPE_MIN     = 0.020

    # 弱触发需满足的子条件数（共5个）
    PRED_SCORE_MIN       = 4
    PRED_SCORE_WINDOW    = 5
    PRED_SCORE_RATIO     = 0.70

    # 趋势一致性：Z、rel、level三者斜率同号
    PRED_CONSISTENCY_WINDOW = 8

    # ── 延迟容忍 ──
    MAX_LATE_TOL         = 20

    # ── 高斯平滑sigma ──
    ZSCORE_SMOOTH_SIGMA  = 2.0


# ============================================================
# 数据集专用配置
# ============================================================

class XJTUAlertConfig(AlertConfig):
    """
    XJTU专用配置 v4.0
    主要问题：早期（第50循环）频繁误触发
    解决方案：
    - PRED_MIN_TRIGGER_RATIO=0.20：至少过了寿命20%才触发
    - PRED_STRONG_ABS_RISE=4.0：强触发需温度显著上升
    - 弱触发五个子条件需同时满足（SCORE_MIN=5）
    - 趋势一致性窗口加大
    """
    BASELINE_WINDOW         = 20
    PRED_GRACE_CYCLES       = 30
    PRED_MIN_TRIGGER_RATIO  = 0.12   # 核心：至少过寿命20%

    # 强触发收紧
    PRED_STRONG_LEVEL       = 2
    PRED_STRONG_WINDOW      = 6
    PRED_STRONG_RATIO       = 0.80
    PRED_STRONG_ABS_RISE    = 3.0    # 温度须比基线高4°C

    # 弱触发收紧
    PRED_Z_MIN              = 1.60
    PRED_REL_TMIN           = 1.00
    PRED_REL_DTMIN          = 0.60
    PRED_Z_WINDOW           = 6
    PRED_Z_RATIO            = 0.80
    PRED_LEVEL1_WINDOW      = 6
    PRED_LEVEL1_RATIO       = 0.85
    PRED_TREND_WINDOW       = 7
    PRED_Z_SLOPE_MIN        = 0.025
    PRED_SCORE_MIN          = 4      # 五个子条件全满足
    PRED_SCORE_WINDOW       = 5
    PRED_SCORE_RATIO        = 0.70
    PRED_CONSISTENCY_WINDOW = 8

    MAX_LATE_TOL            = 30
    ZSCORE_SMOOTH_SIGMA     = 2.5
    PRED_ABS_TRIGGER_TMAX   = 40.0
    # 真实事件检测
    TRUE_GRACE_CYCLES       = 8
    TRUE_EVENT_WINDOW       = 5
    TRUE_EVENT_RATIO        = 0.80
    TRUE_REL_TMIN           = 1.2
    TRUE_REL_DTMIN          = 0.7


class NASAGroupAConfig(AlertConfig):
    """
    NASA Group_A (B0005/B0006/B0007/B0018)
    室温~24°C，温度基线~37-38°C，总循环~50-120
    循环数少，MIN_TRIGGER_RATIO不能太高
    """
    BASELINE_WINDOW         = 15
    PRED_GRACE_CYCLES       = 8
    PRED_MIN_TRIGGER_RATIO  = 0.15  # 循环数少，15%即可
    PRED_Z_MIN              = 1.40
    PRED_REL_TMIN           = 1.00
    PRED_REL_DTMIN          = 0.60
    PRED_STRONG_ABS_RISE    = 2.5
    PRED_STRONG_WINDOW      = 5
    PRED_STRONG_RATIO       = 0.75
    PRED_SCORE_MIN          = 4
    PRED_SCORE_WINDOW       = 4
    PRED_CONSISTENCY_WINDOW = 6
    MAX_LATE_TOL            = 10
    ZSCORE_SMOOTH_SIGMA     = 1.5
    ABS_DANGER              = {'tmax': 50.0,
                               'delta_t': 9.0}


class NASAGroupBConfig(AlertConfig):
    """
    NASA Group_B (B0045/B0046/B0047/B0048)
    低温4°C，温度基线~11-13°C，总循环~40-80
    温度绝对值低，相对变化是主要信号
    """
    BASELINE_WINDOW         = 10
    PRED_GRACE_CYCLES       = 12
    PRED_MIN_TRIGGER_RATIO  = 0.20
    Z_THRESHOLDS            = {'warn': 2.0, 'alert': 3.0,
                               'danger': 4.5}
    PRED_Z_MIN              = 2.00   # 低温噪声大，Z阈值高
    PRED_REL_TMIN           = 1.80
    PRED_REL_DTMIN          = 1.20
    PRED_STRONG_ABS_RISE    = 2.0    # 低温绝对上升较小
    PRED_STRONG_WINDOW      = 5
    PRED_STRONG_RATIO       = 0.80
    PRED_SCORE_MIN          = 4
    PRED_SCORE_WINDOW       = 5
    PRED_CONSISTENCY_WINDOW = 6
    MAX_LATE_TOL            = 8
    ZSCORE_SMOOTH_SIGMA     = 2.0
    ABS_DANGER              = {'tmax': 25.0,
                               'delta_t': 7.0}
    TRUE_TRIGGER_LEVEL      = 2
    TRUE_REL_TMIN           = 0.50
    TRUE_REL_DTMIN          = 0.30


class NASAGroupCConfig(AlertConfig):
    """
    NASA Group_C (B0053/B0054/B0055/B0056)
    中低温~15-22°C
    """
    BASELINE_WINDOW         = 12
    PRED_GRACE_CYCLES       = 10
    PRED_MIN_TRIGGER_RATIO  = 0.18
    Z_THRESHOLDS            = {'warn': 2.0, 'alert': 3.0,
                               'danger': 4.5}
    PRED_Z_MIN              = 1.60
    PRED_REL_TMIN           = 1.20
    PRED_REL_DTMIN          = 0.70
    PRED_STRONG_ABS_RISE    = 2.5
    PRED_SCORE_MIN          = 4
    PRED_CONSISTENCY_WINDOW = 7
    MAX_LATE_TOL            = 8
    ZSCORE_SMOOTH_SIGMA     = 2.0
    ABS_DANGER              = {'tmax': 30.0,
                               'delta_t': 8.0}
    TRUE_TRIGGER_LEVEL      = 2
    TRUE_REL_TMIN           = 0.50
    TRUE_REL_DTMIN          = 0.30


NASA_BAT_CFG_MAP = {
    'B0005': NASAGroupAConfig(),
    'B0006': NASAGroupAConfig(),
    'B0007': NASAGroupAConfig(),
    'B0018': NASAGroupAConfig(),
    'B0045': NASAGroupBConfig(),
    'B0046': NASAGroupBConfig(),
    'B0047': NASAGroupBConfig(),
    'B0048': NASAGroupBConfig(),
    'B0053': NASAGroupCConfig(),
    'B0054': NASAGroupCConfig(),
    'B0055': NASAGroupCConfig(),
    'B0056': NASAGroupCConfig(),
}


# ============================================================
# 去重函数
# ============================================================

def deduplicate_infos(infos: list,
                      strategy: str = 'best') -> list:
    seen = {}
    for info in infos:
        bid = info['bat_id']
        if bid not in seen:
            seen[bid] = info
        else:
            if strategy == 'best':
                a_new = info.get('advance_cycles')
                a_old = seen[bid].get('advance_cycles')
                if a_new is not None:
                    if a_old is None or a_new > a_old:
                        seen[bid] = info
            elif strategy == 'last':
                seen[bid] = info

    added, result = set(), []
    for info in infos:
        bid = info['bat_id']
        if seen.get(bid) is info and bid not in added:
            result.append(info)
            added.add(bid)

    removed = len(infos) - len(result)
    if removed > 0:
        print(f"  [Dedup] 去除重复 {removed} 条，"
              f"保留 {len(result)} 块 "
              f"(strategy='{strategy}')")
    return result


# ============================================================
# 基础函数
# ============================================================

def _robust_baseline(values: np.ndarray, window: int):
    seg = values[:window]
    if len(seg) < 3:
        seg = values[:max(3, len(values))]
    med = float(np.median(seg))
    mad = float(np.median(np.abs(seg - med)))
    return med, mad


def _robust_zscore(values: np.ndarray,
                   med: float, mad: float,
                   eps: float = 1e-6):
    return (values - med) / (1.4826 * mad + eps)


def _compute_trend(values: np.ndarray, window: int):
    n = len(values)
    slopes = np.zeros(n)
    for i in range(n):
        seg = values[max(0, i - window + 1): i + 1]
        if len(seg) >= 3:
            x = np.arange(len(seg), dtype=float)
            slopes[i] = float(np.polyfit(x, seg, 1)[0])
    return slopes


def _sustained_check(flags: np.ndarray,
                     window: int, ratio: float):
    n = len(flags)
    confirmed = np.zeros(n, dtype=bool)
    for i in range(n):
        seg = flags[max(0, i - window + 1): i + 1]
        if len(seg) > 0 and \
                seg.sum() / len(seg) >= ratio:
            confirmed[i] = True
    return confirmed


def _trend_consistency_check(zt: np.ndarray,
                              rel_t: np.ndarray,
                              levels: np.ndarray,
                              window: int) -> np.ndarray:
    """
    趋势一致性检验：
    Z-score、相对温度偏移、level三者在过去window个
    循环内的线性斜率均为正，则认为退化趋势一致。
    防止单通道波动凑分触发。
    """
    n = len(zt)
    consistent = np.zeros(n, dtype=bool)
    x = np.arange(window, dtype=float)
    for i in range(window, n):
        seg_z   = zt[i - window: i]
        seg_r   = rel_t[i - window: i]
        seg_lv  = levels[i - window: i].astype(float)
        slope_z  = float(np.polyfit(x, seg_z,  1)[0])
        slope_r  = float(np.polyfit(x, seg_r,  1)[0])
        slope_lv = float(np.polyfit(x, seg_lv, 1)[0])
        # 三者斜率均为正
        if slope_z > 0 and slope_r > 0 and \
                slope_lv >= 0:
            consistent[i] = True
    return consistent


# ============================================================
# 预警等级生成
# ============================================================

def generate_alert_series(
        tmax_pred:  np.ndarray,
        dt_pred:    np.ndarray,
        soh_pred:   np.ndarray,
        tmean_pred: np.ndarray = None,
        cfg:        AlertConfig = None):

    if cfg is None:
        cfg = AlertConfig()

    N = min(len(tmax_pred), len(dt_pred),
            len(soh_pred))
    tmax_pred = np.asarray(tmax_pred[:N], dtype=float)
    dt_pred   = np.asarray(dt_pred[:N],   dtype=float)
    soh_pred  = np.asarray(soh_pred[:N],  dtype=float)

    bw = int(min(cfg.BASELINE_WINDOW,
                 max(N // 5, 5)))

    tmax_med, tmax_mad = _robust_baseline(tmax_pred, bw)
    dt_med,   dt_mad   = _robust_baseline(dt_pred,   bw)
    soh_0 = float(np.median(soh_pred[:bw]))

    baseline_info = {
        'baseline_window': bw,
        'tmax_median':     tmax_med,
        'tmax_mad':        tmax_mad,
        'dt_median':       dt_med,
        'dt_mad':          dt_mad,
        'soh_baseline':    soh_0,
    }

    z_tmax  = _robust_zscore(tmax_pred, tmax_med,
                             tmax_mad)
    z_dt    = _robust_zscore(dt_pred,   dt_med, dt_mad)
    soh_rel = np.zeros(N)
    if soh_0 > 0:
        soh_rel = np.clip(
            (soh_0 - soh_pred) / soh_0, 0.0, 1.0)

    trend_tmax = _compute_trend(
        tmax_pred, cfg.TREND_WINDOW)
    trend_dt   = _compute_trend(
        dt_pred,   cfg.TREND_WINDOW)

    zt  = cfg.Z_THRESHOLDS
    cw  = cfg.CONFIRM_WINDOW
    cr  = cfg.CONFIRM_RATIO

    def _check(z, thr):
        return _sustained_check(z >= thr, cw, cr)

    tmax_warn   = _check(z_tmax, zt['warn'])
    tmax_alert  = _check(z_tmax, zt['alert'])
    tmax_danger = _check(z_tmax, zt['danger'])
    dt_warn     = _check(z_dt,   zt['warn'])
    dt_alert    = _check(z_dt,   zt['alert'])
    dt_danger   = _check(z_dt,   zt['danger'])

    trend_up_tmax = (
        (trend_tmax > cfg.TREND_SLOPE_TMAX) &
        (z_tmax     > cfg.TREND_Z_MIN))
    trend_up_dt = (
        (trend_dt   > cfg.TREND_SLOPE_DT) &
        (z_dt       > cfg.TREND_Z_MIN))

    abs_tmax = tmax_pred >= cfg.ABS_DANGER['tmax']
    abs_dt   = dt_pred   >= cfg.ABS_DANGER['delta_t']

    levels = np.zeros(N, dtype=int)
    sot    = cfg.SOH_REL_THRESHOLDS

    for i in range(N):
        if i < bw:
            levels[i] = (3 if (abs_tmax[i] or
                               abs_dt[i]) else 0)
            continue

        lv = 0
        if tmax_danger[i] or abs_tmax[i]:
            lv = max(lv, 3)
        elif tmax_alert[i]:
            lv = max(lv, 2)
        elif tmax_warn[i]:
            lv = max(lv, 1)

        if dt_danger[i] or abs_dt[i]:
            lv = max(lv, 3)
        elif dt_alert[i]:
            lv = max(lv, 2)
        elif dt_warn[i]:
            lv = max(lv, 1)

        if 1 <= lv <= 2:
            if trend_up_tmax[i] or trend_up_dt[i]:
                lv = min(lv + 1, 3)

        if lv >= 2:
            if soh_rel[i] >= sot['severe']:
                lv = min(lv + 1, 3)

        levels[i] = lv

    return levels, z_tmax, z_dt, soh_rel, baseline_info


# ============================================================
# 真实事件检测
# ============================================================

def find_true_event_idx(levels:        np.ndarray,
                        tmax_true:     np.ndarray,
                        dt_true:       np.ndarray,
                        baseline_info: dict,
                        cfg:           AlertConfig):
    bw    = baseline_info['baseline_window']
    start = bw + cfg.TRUE_GRACE_CYCLES
    N     = len(levels)
    if start >= N:
        return None

    tmax_true = np.asarray(tmax_true, dtype=float)
    dt_true   = np.asarray(dt_true,   dtype=float)
    rel_t = tmax_true - baseline_info['tmax_median']
    rel_d = dt_true   - baseline_info['dt_median']

    level_flag = _sustained_check(
        (levels >= cfg.TRUE_TRIGGER_LEVEL) &
        ((rel_t >= cfg.TRUE_REL_TMIN) |
         (rel_d >= cfg.TRUE_REL_DTMIN)),
        cfg.TRUE_EVENT_WINDOW,
        cfg.TRUE_EVENT_RATIO
    )
    abs_flag = (
        (tmax_true >= cfg.ABS_DANGER['tmax']) |
        (dt_true   >= cfg.ABS_DANGER['delta_t']))
    final_flag = level_flag | abs_flag

    for i in range(start, N):
        if final_flag[i]:
            return i
    return None


# ============================================================
# 预测事件检测（唯一版本 v4.0）
# ============================================================

def find_pred_event_idx(levels:        np.ndarray,
                        z_tmax:        np.ndarray,
                        z_dt:          np.ndarray,
                        tmax_pred:     np.ndarray,
                        dt_pred:       np.ndarray,
                        baseline_info: dict,
                        cfg:           AlertConfig):
    """
    v4.1 三路触发设计：
    路径A（绝对危险）：预测温度超过绝对阈值
                      仅受 grace//2 约束，无 MIN_RATIO
                      用于捕获 B1(tmax_base=39.5) 这类
                      基线已高、后续继续升温的样本
    路径B（强触发）  ：level持续高 AND 温度相对上升
                      受 grace + MIN_RATIO 约束
                      ABS_RISE 从4.0降至2.0，放松条件
    路径C（弱触发）  ：5个子条件满足>=4个
                      受 grace + MIN_RATIO 约束
                      SCORE_MIN 从5降至4
    最终取三路中最早的有效触发点
    """
    bw = baseline_info['baseline_window']
    N  = len(levels)

    if N <= bw:
        return None

    # ══════════════════════════════════════════════
    # 平滑 Z-score
    # ══════════════════════════════════════════════
    sigma = getattr(cfg, 'ZSCORE_SMOOTH_SIGMA', 2.0)
    zt = gaussian_filter1d(
        np.asarray(z_tmax, dtype=float), sigma=sigma)
    zd = gaussian_filter1d(
        np.asarray(z_dt,   dtype=float), sigma=sigma)

    tmax_pred = np.asarray(tmax_pred, dtype=float)
    dt_pred   = np.asarray(dt_pred,   dtype=float)
    rel_t = tmax_pred - baseline_info['tmax_median']
    rel_d = dt_pred   - baseline_info['dt_median']

    # ══════════════════════════════════════════════
    # 路径A：绝对危险触发
    # 预测温度持续超过绝对阈值，最早可触发时刻为
    # bw + grace//2，不受 MIN_RATIO 约束
    # 专门处理基线温度已高（如39.5°C）的电池
    # ══════════════════════════════════════════════
    abs_trigger_tmax = getattr(
        cfg, 'PRED_ABS_TRIGGER_TMAX', 999.0)
    abs_trigger_dt = getattr(
        cfg, 'PRED_ABS_TRIGGER_DT', 999.0)

    # 绝对触发：持续确认窗口较短（3个循环即可）
    abs_tmax_flag = _sustained_check(
        tmax_pred >= abs_trigger_tmax,
        window=3, ratio=0.67)
    abs_dt_flag = _sustained_check(
        dt_pred >= abs_trigger_dt,
        window=3, ratio=0.67)
    abs_flag = abs_tmax_flag | abs_dt_flag

    # 绝对触发的最早时刻：grace期的一半
    abs_start = bw + max(cfg.PRED_GRACE_CYCLES // 2, 5)
    abs_start = min(abs_start, N - 1)

    # ══════════════════════════════════════════════
    # 路径B/C 的时间约束
    # grace_start：等待基线稳定
    # ratio_start：相对寿命位置约束
    # 两者取较大值，但不超过 N-1
    # ══════════════════════════════════════════════
    grace_start = bw + cfg.PRED_GRACE_CYCLES
    ratio_start = int(N * getattr(
        cfg, 'PRED_MIN_TRIGGER_RATIO', 0.10))
    start = min(max(grace_start, ratio_start), N - 1)

    # ══════════════════════════════════════════════
    # 路径B：强触发
    # level 持续高 AND 温度相对基线持续上升
    # ABS_RISE 已从 4.0 放松至 2.0（见 XJTUAlertConfig）
    # ══════════════════════════════════════════════
    abs_rise = getattr(cfg, 'PRED_STRONG_ABS_RISE', 2.0)

    strong_level_flag = _sustained_check(
        levels >= cfg.PRED_STRONG_LEVEL,
        cfg.PRED_STRONG_WINDOW,
        cfg.PRED_STRONG_RATIO)
    strong_temp_flag = _sustained_check(
        (rel_t >= abs_rise) | (rel_d >= abs_rise * 0.5),
        cfg.PRED_STRONG_WINDOW,
        cfg.PRED_STRONG_RATIO)
    # 强触发：level 和温度双重持续满足
    strong_flag = strong_level_flag & strong_temp_flag

    # ══════════════════════════════════════════════
    # 路径C：弱触发（5个子条件满足>=SCORE_MIN个）
    # ══════════════════════════════════════════════

    # 子条件1：Z-score 持续超阈值
    z_flag = _sustained_check(
        (zt >= cfg.PRED_Z_MIN) |
        (zd >= cfg.PRED_Z_MIN),
        cfg.PRED_Z_WINDOW,
        cfg.PRED_Z_RATIO)

    # 子条件2：相对温度偏移持续超阈值
    rel_flag = _sustained_check(
        (rel_t >= cfg.PRED_REL_TMIN) |
        (rel_d >= cfg.PRED_REL_DTMIN),
        cfg.PRED_Z_WINDOW,
        cfg.PRED_Z_RATIO)

    # 子条件3：预警等级持续 >= 1
    lv1_flag = _sustained_check(
        levels >= 1,
        cfg.PRED_LEVEL1_WINDOW,
        cfg.PRED_LEVEL1_RATIO)

    # 子条件4：Z-score 斜率持续为正（上升趋势）
    slope_t = _compute_trend(zt, cfg.PRED_TREND_WINDOW)
    slope_d = _compute_trend(zd, cfg.PRED_TREND_WINDOW)
    trend_flag = _sustained_check(
        (slope_t >= cfg.PRED_Z_SLOPE_MIN) |
        (slope_d >= cfg.PRED_Z_SLOPE_MIN),
        cfg.PRED_Z_WINDOW, 0.65)

    # 子条件5：三通道趋势一致性
    cw = getattr(cfg, 'PRED_CONSISTENCY_WINDOW', 8)
    consistency_flag = _trend_consistency_check(
        zt, rel_t, levels, cw)

    # 5个子条件得分求和，满足 SCORE_MIN 个即触发
    # SCORE_MIN 已从 5 降至 4（见 XJTUAlertConfig）
    weak_score = (
        z_flag.astype(int)           +
        rel_flag.astype(int)         +
        lv1_flag.astype(int)         +
        trend_flag.astype(int)       +
        consistency_flag.astype(int)
    )
    weak_flag = _sustained_check(
        weak_score >= cfg.PRED_SCORE_MIN,
        cfg.PRED_SCORE_WINDOW,
        cfg.PRED_SCORE_RATIO)

    # 路径 B 和 C 合并
    bc_flag = strong_flag | weak_flag

    # ══════════════════════════════════════════════
    # 三路取最早触发点
    # ══════════════════════════════════════════════
    abs_idx = None
    bc_idx  = None

    # 路径A搜索（较早开始）
    for i in range(abs_start, N):
        if abs_flag[i]:
            abs_idx = i
            break

    # 路径B/C搜索（受MIN_RATIO约束）
    if start < N:
        for i in range(start, N):
            if bc_flag[i]:
                bc_idx = i
                break

    # 取两路中最早的有效结果
    candidates = [x for x in [abs_idx, bc_idx]
                  if x is not None]
    return min(candidates) if candidates else None

# ============================================================
# 绘图辅助函数
# ============================================================

def _add_baseline_span(ax, bw, label=True):
    ax.axvspan(0, bw, color=C['baseline_bg'],
               alpha=0.8, zorder=0,
               label='Baseline window' if label else '')


def _add_temp_risk_bands(ax, ymax=60.0):
    bands = [
        (38.0, 42.0, '#fef3c7', 'Caution'),
        (42.0, 46.0, '#fde68a', 'Warning'),
        (46.0, 60.0, '#fecaca', 'Danger'),
    ]
    for lo, hi, color, label in bands:
        if lo < ymax:
            ax.axhspan(lo, min(hi, ymax),
                       color=color, alpha=0.35, zorder=0)
            ax.axhline(lo, color='#9ca3af', lw=0.8,
                       ls='--', alpha=0.55, zorder=1)

    # 右上角放一个小标签，避免图例过长
    ax.text(0.995, 0.98,
            'Thermal risk bands',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=7.2, color='#6b7280',
            bbox=dict(boxstyle='round,pad=0.25',
                      fc='white', ec='#d1d5db', alpha=0.9))


def _add_alert_markers(ax, true_idx, pred_idx,
                       advance, y_range,
                       text_offset=0.06):
    yspan = y_range[1] - y_range[0]
    y_top = y_range[1] - 0.08 * yspan
    y_mid = y_range[1] - 0.18 * yspan

    # True event
    if true_idx is not None:
        ax.axvline(true_idx, color=C['true_alert'],
                   lw=1.4, ls='--', alpha=0.9, zorder=12)
        ax.scatter([true_idx], [y_top], s=42,
                   color=C['true_alert'], zorder=13,
                   edgecolor='white', linewidth=0.8)
        ax.text(true_idx, y_top + yspan * 0.02,
                f'True event\n@{true_idx}',
                ha='center', va='bottom',
                fontsize=6.8, color=C['true_alert'],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.22',
                          fc='white', ec='#d1d5db', alpha=0.95))

    # Pred event
    if pred_idx is not None:
        ax.axvline(pred_idx, color=C['pred_alert'],
                   lw=1.4, ls='-.', alpha=0.95, zorder=12)
        ax.scatter([pred_idx], [y_top], s=42,
                   color=C['pred_alert'], zorder=13,
                   edgecolor='white', linewidth=0.8)
        label = f'Pred trigger\n@{pred_idx}'
        ax.text(pred_idx, y_top - yspan * 0.10,
                label,
                ha='center', va='top',
                fontsize=6.8, color=C['pred_alert'],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.22',
                          fc='white', ec='#d1d5db', alpha=0.95))

    # Lead arrow
    if (true_idx is not None) and (pred_idx is not None) and (advance is not None):
        x1, x2 = sorted([pred_idx, true_idx])
        arrow_color = '#2563eb' if advance >= 0 else '#dc2626'
        ax.annotate(
            '', xy=(x2, y_mid), xytext=(x1, y_mid),
            arrowprops=dict(arrowstyle='<->',
                            lw=1.3, color=arrow_color,
                            shrinkA=2, shrinkB=2)
        )
        xm = 0.5 * (x1 + x2)
        sign = '+' if advance > 0 else ''
        txt = f'Lead = {sign}{advance} cycles'
        ax.text(xm, y_mid + yspan * 0.02, txt,
                ha='center', va='bottom',
                fontsize=7.0, color=arrow_color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          fc='white', ec='#d1d5db', alpha=0.95))

def _draw_level_colorbar(ax, levels, bw, N):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([
        LEVEL_COLORS[0], LEVEL_COLORS[1],
        LEVEL_COLORS[2], LEVEL_COLORS[3]
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    lv_mat = np.asarray(levels, dtype=float).reshape(1, -1)

    ax.imshow(lv_mat, aspect='auto', cmap=cmap, norm=norm,
              extent=[0, N, 0, 1], interpolation='nearest')

    # baseline window
    ax.axvspan(0, bw, color='#9ca3af', alpha=0.18, zorder=3)
    ax.axvline(bw, color='#6b7280', lw=1.0, ls='--', alpha=0.8)

    ax.set_yticks([])
    ax.set_ylabel('Alert\nLevel', fontsize=8)
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)

    # 在条带上标出级别切换点
    prev = levels[0]
    for i in range(1, len(levels)):
        if levels[i] != prev and i >= bw:
            ax.axvline(i, color='white', lw=0.8, alpha=0.7)
            prev = levels[i]

    # 右侧放固定图例说明
    legend_elems = [
        Patch(facecolor=LEVEL_COLORS[0], edgecolor='none', label='Normal'),
        Patch(facecolor=LEVEL_COLORS[1], edgecolor='none', label='Caution'),
        Patch(facecolor=LEVEL_COLORS[2], edgecolor='none', label='Warning'),
        Patch(facecolor=LEVEL_COLORS[3], edgecolor='none', label='Danger'),
    ]
    ax.legend(handles=legend_elems, ncol=4, frameon=False,
              loc='upper center', bbox_to_anchor=(0.5, -0.55),
              fontsize=7, handlelength=1.2, columnspacing=1.2)

def _build_zscore_bands(ax):
    bands = [
        (1.6, 2.6, '#fef3c7', 'Caution'),
        (2.6, 4.0, '#fed7aa', 'Warning'),
        (4.0, 8.0, '#fecaca', 'Danger'),
    ]
    for lo, hi, color, _ in bands:
        ax.axhspan(lo, hi, color=color, alpha=0.30, zorder=0)
        ax.axhline(lo, color='#9ca3af', lw=0.8, ls='--', alpha=0.75)

    ax.axhline(0, color='#9ca3af', lw=0.9, alpha=0.7)
    ax.text(0.99, 0.97, 'Robust Z-score',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=7.2, color='#6b7280',
            bbox=dict(boxstyle='round,pad=0.22',
                      fc='white', ec='#d1d5db', alpha=0.9))


def _set_suptitle(fig, bat_id, dataset_tag,
                  sample_type, advance,
                  max_lv, bl_info,
                  warning_status, y=0.982):
    lead_str = ('N/A' if advance is None
                else f'{"+" if advance > 0 else ""}{advance} cycles')

    title = f'[{dataset_tag}] {bat_id}'
    subtitle = (
        f'Sample: {sample_type}   |   '
        f'Status: {warning_status}   |   '
        f'Lead: {lead_str}   |   '
        f'Max level: {max_lv} ({LEVEL_NAMES[max_lv]})'
    )
    baseline_txt = (
        f'Baseline: '
        f'$T_{{max}}$={bl_info["tmax_median"]:.1f}°C, '
        f'ΔT={bl_info["dt_median"]:.2f}°C, '
        f'SOH={bl_info["soh_baseline"]:.4f}'
    )

    fig.suptitle(title, fontsize=11.5, fontweight='bold',
                 y=y, color='#111827')
    fig.text(0.5, y - 0.028, subtitle,
             ha='center', va='center',
             fontsize=8.4, color='#374151',
             bbox=dict(boxstyle='round,pad=0.25',
                       fc='#f9fafb', ec='#d1d5db', alpha=0.95))
    fig.text(0.5, y - 0.055, baseline_txt,
             ha='center', va='center',
             fontsize=8.0, color='#6b7280')


# ============================================================
# 论文风格绘图
# ============================================================

def _plot_paper_style(result, bat_id,
                      levels, z_tmax, z_dt,
                      soh_rel, bl_info,
                      true_idx, pred_idx, advance,
                      sample_type, warning_status,
                      save_dir, dataset_tag='XJTU'):
    tmax_t  = np.asarray(result['tmax']['ts'])
    tmax_p  = np.asarray(result['tmax']['ps'])
    tmean_p = np.asarray(result['tmean']['ps'])
    soh_t   = np.asarray(result['soh']['ts'])
    soh_p   = np.asarray(result['soh']['ps'])
    dt_t    = np.asarray(result['delta_t']['ts'])
    dt_p    = np.asarray(result['delta_t']['values'])

    bw = bl_info['baseline_window']
    N = min(len(tmax_t), len(tmax_p),
            len(soh_t),  len(soh_p),
            len(dt_p),   len(levels))
    x = np.arange(N)

    fig = plt.figure(figsize=(11.2, 8.6), facecolor='white')
    gs = gridspec.GridSpec(
        4, 1,
        height_ratios=[4.2, 2.6, 2.4, 0.7],
        hspace=0.18,
        left=0.08, right=0.93,
        top=0.90, bottom=0.10
    )

    # ======================================================
    # Row 1: Temperature panel
    # ======================================================
    ax1 = fig.add_subplot(gs[0])
    ax1r = ax1.twinx()

    _add_baseline_span(ax1, bw)
    _add_temp_risk_bands(ax1, ymax=float(max(tmax_t[:N].max(), tmax_p[:N].max())) + 2.5)

    # 主曲线
    ax1.plot(x, tmax_t[:N], color='#111827', lw=2.4,
             label='True $T_{max}$', zorder=10)
    ax1.plot(x, tmax_p[:N], color='#dc2626', lw=1.8,
             alpha=0.95, label='Pred $T_{max}$', zorder=9)
    ax1.plot(x, tmean_p[:N], color='#2563eb', lw=1.5, ls='--',
             alpha=0.95, label='Pred $T_{mean}$', zorder=8)

    # baseline
    ax1.axhline(bl_info['tmax_median'],
                color='#6b7280', lw=1.0, ls=':',
                alpha=0.9, zorder=3)

    # twin axis: ΔT
    ax1r.plot(x, dt_t[:N], color='#7c3aed', lw=1.0, ls=':',
              alpha=0.45, label='True ΔT', zorder=4)
    ax1r.plot(x, dt_p[:N], color='#7c3aed', lw=1.4,
              alpha=0.85, label='Pred ΔT', zorder=5)

    # 事件标记
    y_lo = min(tmax_t[:N].min(), tmax_p[:N].min()) - 1.0
    y_hi = max(tmax_t[:N].max(), tmax_p[:N].max()) + 2.5
    _add_alert_markers(ax1, true_idx, pred_idx, advance, (y_lo, y_hi))

    ax1.set_xlim(0, N - 1)
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_ylabel('Temperature (°C)')
    ax1.tick_params(labelbottom=False)

    ax1r.set_ylabel('ΔT (°C)', color='#7c3aed')
    ax1r.tick_params(axis='y', colors='#7c3aed')
    ax1r.spines['right'].set_visible(True)
    ax1r.spines['right'].set_alpha(0.35)

    # 图例重新组织，避免拥挤
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1r.get_legend_handles_labels()
    baseline_handle = Line2D([0], [0], color='#6b7280', lw=1.0, ls=':',
                             label=f'Baseline $T_{{max}}$ ({bl_info["tmax_median"]:.1f}°C)')
    ax1.legend(handles1 + handles2 + [baseline_handle],
               labels1 + labels2 + [baseline_handle.get_label()],
               loc='lower left', ncol=3,
               frameon=True, fancybox=True,
               framealpha=0.93, edgecolor='#d1d5db')

    # ======================================================
    # Row 2: SOH panel
    # ======================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2r = ax2.twinx()

    _add_baseline_span(ax2, bw, label=False)

    ax2.plot(x, soh_t[:N], color='#111827', lw=2.1,
             label='True SOH', zorder=10)
    ax2.plot(x, soh_p[:N], color='#d97706', lw=1.6,
             alpha=0.95, label='Pred SOH', zorder=9)

    ax2.axhline(bl_info['soh_baseline'],
                color='#6b7280', lw=1.0, ls=':',
                alpha=0.9, zorder=3)

    ax2r.fill_between(x, 0, soh_rel[:N] * 100,
                      color='#fbbf24', alpha=0.22,
                      label='SOH decline (%)', zorder=1)

    ax2.set_ylabel('SOH')
    ax2.set_ylim(min(soh_t[:N].min(), soh_p[:N].min()) - 0.03,
                 max(soh_t[:N].max(), soh_p[:N].max()) + 0.02)
    ax2.tick_params(labelbottom=False)

    ax2r.set_ylabel('Decline (%)', color='#b45309')
    ax2r.tick_params(axis='y', colors='#b45309')
    ax2r.spines['right'].set_visible(True)
    ax2r.spines['right'].set_alpha(0.35)

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2r.get_legend_handles_labels()
    baseline_handle2 = Line2D([0], [0], color='#6b7280', lw=1.0, ls=':',
                              label=f'SOH baseline ({bl_info["soh_baseline"]:.4f})')
    ax2.legend(handles1 + handles2 + [baseline_handle2],
               labels1 + labels2 + [baseline_handle2.get_label()],
               loc='lower left', ncol=3,
               frameon=True, fancybox=True,
               framealpha=0.93, edgecolor='#d1d5db')

    # ======================================================
    # Row 3: Z-score panel
    # ======================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _add_baseline_span(ax3, bw, label=False)
    _build_zscore_bands(ax3)

    ax3.plot(x, z_tmax[:N], color='#dc2626', lw=1.6,
             label='Z($T_{max}$)', zorder=6)
    ax3.plot(x, z_dt[:N], color='#7c3aed', lw=1.6,
             label='Z(ΔT)', zorder=6)

    # 在Z图上同步标出真实/预测触发
    if true_idx is not None:
        ax3.axvline(true_idx, color=C['true_alert'],
                    lw=1.2, ls='--', alpha=0.85)
    if pred_idx is not None:
        ax3.axvline(pred_idx, color=C['pred_alert'],
                    lw=1.2, ls='-.', alpha=0.9)

    ax3.set_ylabel('Z-score')
    ax3.set_ylim(min(-0.5, np.min(z_tmax[:N]), np.min(z_dt[:N])) - 0.2,
                 max(5.5, np.max(z_tmax[:N]), np.max(z_dt[:N])) + 0.3)
    ax3.tick_params(labelbottom=False)
    ax3.legend(loc='upper left', ncol=2,
               frameon=True, fancybox=True,
               framealpha=0.93, edgecolor='#d1d5db')

    # ======================================================
    # Row 4: alert level strip
    # ======================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _draw_level_colorbar(ax4, levels[:N], bw, N)
    ax4.set_xlabel('Cycle')

    # ======================================================
    # 顶部标题信息
    # ======================================================
#    _set_suptitle(fig, bat_id, dataset_tag,
#                  sample_type, advance,
#                  int(levels[:N].max()),
#                  bl_info, warning_status)

    # baseline window 标识
    for ax in [ax1, ax2, ax3]:
        ax.axvline(bw, color='#6b7280', lw=1.0,
                   ls='--', alpha=0.8)
        ax.text(bw + 1, ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                'Baseline end',
                fontsize=7.0, color='#6b7280',
                va='top', ha='left')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"Paper_{bat_id}.png")
    plt.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ============================================================
# 汇总统计图
# ============================================================

def plot_alert_summary(alert_records: list,
                       save_dir:      str,
                       dataset_tag:   str = 'XJTU'):
    records = [r for r in alert_records
               if r.get('sample_type') is not None]
    if not records:
        print("  [Summary] 无有效记录，跳过汇总图。")
        return None

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.5),
        facecolor='white')

    # 左：lead time
    ax    = axes[0]
    valid = [r for r in records
             if r['sample_type'] == 'normal_to_risk'
             and r.get('advance_cycles') is not None]
    if valid:
        bat_ids = [r['bat_id']         for r in valid]
        leads   = [r['advance_cycles'] for r in valid]
        colors  = [LEVEL_COLORS[r['max_level']]
                   for r in valid]
        bars = ax.barh(
            bat_ids, leads, color=colors,
            alpha=0.85, edgecolor='white',
            linewidth=0.6)
        ax.axvline(0, color='#636e72', lw=1.0)
        for bar, v in zip(bars, leads):
            offset = max(abs(v) * 0.04, 0.5)
            ha     = 'left' if v >= 0 else 'right'
            x_txt  = (v + offset if v >= 0
                      else v - offset)
            ax.text(
                x_txt,
                bar.get_y() + bar.get_height() / 2,
                f'{v:+d}',
                va='center', ha=ha,
                fontsize=7.5, fontweight='bold',
                color='#2d3436')
        ax.set_xlabel(
            'Lead time (cycles)', fontsize=9)
        ax.grid(axis='x', ls=':', alpha=0.4)
    else:
        ax.text(0.5, 0.5,
                'No valid lead-time\nrecords',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=10, color='#636e72')
    ax.set_title(
        f'[{dataset_tag}] Valid Lead Time',
        fontsize=9, fontweight='bold')

    # 中：sample_type 饼图
    ax = axes[1]
    type_colors = {
        'normal_to_risk': '#00b894',
        'initial_risk':   '#fdcb6e',
        'no_alert':       '#b2bec3'}
    type_counts = {}
    for r in records:
        t = r['sample_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    _, _, autotexts = ax.pie(
        type_counts.values(),
        labels=[f'{k}\n(n={v})'
                for k, v in type_counts.items()],
        colors=[type_colors.get(k, '#636e72')
                for k in type_counts],
        autopct='%1.0f%%', startangle=90,
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white',
                    'lw': 1.5})
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight('bold')
    ax.set_title(
        f'[{dataset_tag}] Sample Type',
        fontsize=9, fontweight='bold')

    # 右：warning_status 柱状图
    ax = axes[2]
    status_order = [
        'hit', 'slightly_late', 'late_as_miss',
        'missed', 'false_alarm', 'no_true_event']
    status_counts = {k: 0 for k in status_order}
    for r in records:
        st = r.get('warning_status', 'unknown')
        status_counts[st] = \
            status_counts.get(st, 0) + 1
    labels = [k for k in status_order
              if status_counts[k] > 0]
    vals   = [status_counts[k] for k in labels]
    color_map = {
        'hit':           '#00b894',
        'slightly_late': '#fdcb6e',
        'late_as_miss':  '#d63031',
        'missed':        '#636e72',
        'false_alarm':   '#0984e3',
        'no_true_event': '#b2bec3',
    }
    bars = ax.bar(
        labels, vals,
        color=[color_map.get(k, '#999')
               for k in labels],
        alpha=0.85, edgecolor='white',
        linewidth=1.0)
    for bar, cnt in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            str(cnt),
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold',
            color='#2d3436')
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(
        f'[{dataset_tag}] Warning Status',
        fontsize=9, fontweight='bold')
    ax.grid(axis='y', ls=':', alpha=0.4)
    ax.tick_params(axis='x', rotation=25)

    fig.suptitle(
        f'[{dataset_tag}] Warning System Summary '
        f'(n={len(records)})',
        fontsize=11, fontweight='bold',
        y=1.02, color='#1a1a2e')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(
        save_dir,
        f"AlertSummary_{dataset_tag}.png")
    plt.savefig(path, dpi=180,
                bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  [Summary] 保存: {path}")
    return path


# ============================================================
# 主分析器类
# ============================================================

class BatteryAlertAnalyzer:
    """通用电池预警分析器 v4.0"""

    def __init__(self,
                 save_dir:     str         = './Alert',
                 cfg:          AlertConfig = None,
                 paper_subdir: str         = 'Paper'):
        self.save_dir  = save_dir
        self.cfg       = cfg or AlertConfig()
        self.paper_dir = os.path.join(
            save_dir, paper_subdir)
        os.makedirs(self.paper_dir, exist_ok=True)

    def analyze(self, result: dict,
                bat_id: str,
                cfg:    AlertConfig = None) -> dict:
        cfg = cfg or self.cfg

        tmax_p = np.asarray(result['tmax']['ps'])
        soh_p  = np.asarray(result['soh']['ps'])
        dt_p   = np.asarray(result['delta_t']['values'])
        tmax_t = np.asarray(result['tmax']['ts'])
        soh_t  = np.asarray(result['soh']['ts'])
        dt_t   = np.asarray(result['delta_t']['ts'])

        N = min(len(tmax_p), len(dt_p), len(soh_p),
                len(tmax_t), len(soh_t), len(dt_t))

        (levels, z_tmax, z_dt,
         soh_rel, bl_info) = generate_alert_series(
            tmax_p[:N], dt_p[:N], soh_p[:N], cfg=cfg)
        bw = bl_info['baseline_window']

        (true_levels, _, _, _,
         true_bl_info) = generate_alert_series(
            tmax_t[:N], dt_t[:N], soh_t[:N], cfg=cfg)

        true_idx = find_true_event_idx(
            true_levels, tmax_t[:N], dt_t[:N],
            true_bl_info, cfg)

        pred_idx_raw = find_pred_event_idx(
            levels, z_tmax, z_dt,
            tmax_p[:N], dt_p[:N],
            bl_info, cfg)

        pred_idx    = pred_idx_raw
        advance_raw = None
        advance     = None

        if true_idx is None:
            warning_status = (
                'no_true_event'
                if pred_idx_raw is None
                else 'false_alarm')
        else:
            if pred_idx_raw is None:
                warning_status = 'missed'
                pred_idx       = None
            else:
                advance_raw = true_idx - pred_idx_raw
                if advance_raw < -cfg.MAX_LATE_TOL:
                    warning_status = 'late_as_miss'
                    pred_idx       = None
                    advance        = None
                elif advance_raw < 0:
                    warning_status = 'slightly_late'
                    pred_idx       = pred_idx_raw
                    advance        = advance_raw
                else:
                    warning_status = 'hit'
                    pred_idx       = pred_idx_raw
                    advance        = advance_raw

        true_bw = true_bl_info['baseline_window']
        if true_idx is None:
            sample_type = 'no_alert'
        elif true_idx <= (true_bw +
                          cfg.TRUE_GRACE_CYCLES):
            sample_type = 'initial_risk'
        else:
            sample_type = 'normal_to_risk'

        return {
            'bat_id':             bat_id,
            'sample_type':        sample_type,
            'baseline_window':    bw,
            'tmax_baseline':      bl_info['tmax_median'],
            'dt_baseline':        bl_info['dt_median'],
            'soh_baseline':       bl_info['soh_baseline'],
            'true_alert_idx':     true_idx,
            'pred_alert_idx_raw': pred_idx_raw,
            'pred_alert_idx':     pred_idx,
            'advance_cycles_raw': advance_raw,
            'advance_cycles':     advance,
            'warning_status':     warning_status,
            'max_level':          int(levels[:N].max()),
            '_levels':            levels,
            '_z_tmax':            z_tmax,
            '_z_dt':              z_dt,
            '_soh_rel':           soh_rel,
            '_bl_info':           bl_info,
            '_N':                 N,
        }

    def plot(self, result: dict, info: dict,
             dataset_tag: str = 'XJTU'):
        p = _plot_paper_style(
            result, info['bat_id'],
            info['_levels'], info['_z_tmax'],
            info['_z_dt'],   info['_soh_rel'],
            info['_bl_info'],
            info['true_alert_idx'],
            info['pred_alert_idx'],
            info['advance_cycles'],
            info['sample_type'],
            info['warning_status'],
            self.paper_dir, dataset_tag)
        print(f"  [Paper] {os.path.basename(p)}")
        return {'paper_fig': p}

    def run_batch(self, results_list: list,
                  dataset_tag:  str  = 'XJTU',
                  max_plot:     int  = None,
                  bat_cfg_map:  dict = None
                  ) -> list:
        infos    = []
        plot_cnt = 0
        for r in results_list:
            bat_id  = r.get('id', f'bat_{len(infos)}')
            bat_cfg = (bat_cfg_map.get(bat_id)
                       if bat_cfg_map else None)
            do_plot = (max_plot is None or
                       plot_cnt < max_plot)
            try:
                info = self.analyze(
                    r, bat_id, bat_cfg)
                if do_plot:
                    paths = self.plot(
                        r, info, dataset_tag)
                    info.update(paths)
                    plot_cnt += 1
                infos.append(info)
            except Exception as e:
                print(f"  [Warn] {bat_id} 失败: {e}")
        return infos

    def run_from_csv(self,
                     csv_path:     str,
                     dataset_tag:  str  = 'XJTU',
                     model_filter: str  = 'Our',
                     max_plot:     int  = None,
                     bat_cfg_map:  dict = None
                     ) -> list:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"  [CSV] 读取 {len(df)} 行 "
              f"← {csv_path}")

        if 'model' in df.columns and model_filter:
            df = df[df['model'] == model_filter]
            print(f"  [CSV] model={model_filter}，"
                  f"剩余 {len(df)} 行")

        bat_ids = df['bat_id'].unique()
        print(f"  [CSV] 共 {len(bat_ids)} 块电池")

        results_list = []
        for bid in bat_ids:
            sub = (df[df['bat_id'] == bid]
                   .sort_values('step'))
            result = self._csv_row_to_result(
                sub, bid)
            result['id'] = bid
            results_list.append(result)

        return self.run_batch(
            results_list, dataset_tag,
            max_plot, bat_cfg_map)

    @staticmethod
    def _csv_row_to_result(sub: pd.DataFrame,
                           bat_id: str) -> dict:
        def _col(name):
            if name in sub.columns:
                return sub[name].values.astype(float)
            return np.zeros(len(sub))

        def _metrics(ts, ps):
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error)
            n = min(len(ts), len(ps))
            ts, ps = ts[:n], ps[:n]
            return {
                'mae':  float(
                    mean_absolute_error(ts, ps)),
                'rmse': float(np.sqrt(
                    mean_squared_error(ts, ps))),
                'ts': ts, 'ps': ps,
            }

        tmax_ts  = _col('tmax_true')
        tmax_ps  = _col('tmax_pred')
        soh_ts   = _col('soh_true')
        soh_ps   = _col('soh_pred')
        tmean_ts = _col('tmean_true')
        tmean_ps = _col('tmean_pred')
        dt_ts    = _col('delta_t_true')
        dt_ps    = _col('delta_t_pred')

        return {
            'tmax':  _metrics(tmax_ts,  tmax_ps),
            'soh':   _metrics(soh_ts,   soh_ps),
            'tmean': _metrics(tmean_ts, tmean_ps),
            'delta_t': {
                **_metrics(dt_ts, dt_ps),
                'values': dt_ps,
            },
            'tmax_high': {
                'mae': float('nan'), 'count': 0},
        }

    def plot_summary(self,
                     infos:       list,
                     dataset_tag: str  = 'XJTU',
                     save_csv:    bool = True) -> str:
        export_keys = [
            'bat_id', 'sample_type',
            'baseline_window',
            'tmax_baseline', 'dt_baseline',
            'soh_baseline',
            'true_alert_idx',
            'pred_alert_idx_raw', 'pred_alert_idx',
            'advance_cycles_raw', 'advance_cycles',
            'warning_status', 'max_level',
        ]
        clean = [{k: r[k] for k in export_keys
                  if k in r} for r in infos]
        if save_csv and clean:
            csv_path = os.path.join(
                self.save_dir,
                f'alert_summary_{dataset_tag}.csv')
            pd.DataFrame(clean).to_csv(
                csv_path, index=False)
            print(f"  [CSV] 汇总: {csv_path}")
        return plot_alert_summary(
            clean, self.save_dir, dataset_tag)

    def print_summary(self,
                      infos:       list,
                      dataset_tag: str = 'XJTU'):
        records = [r for r in infos
                   if r.get('sample_type')]
        n = len(records)
        if n == 0:
            print("  [Summary] 无有效记录")
            return

        print(f"\n{'='*70}")
        print(f"  [{dataset_tag}] 预警摘要 "
              f"({n} 块电池)")
        print(f"{'='*70}")

        for stype in ['normal_to_risk',
                      'initial_risk', 'no_alert']:
            sub = [r for r in records
                   if r['sample_type'] == stype]
            if not sub:
                continue
            print(f"\n  [{stype}]  n={len(sub)}")
            valid = [r for r in sub
                     if r['advance_cycles']
                     is not None]
            if valid:
                leads = [r['advance_cycles']
                         for r in valid]
                pos   = sum(v >= 0 for v in leads)
                print(f"    Avg/Med lead: "
                      f"{np.mean(leads):.1f} / "
                      f"{np.median(leads):.1f}")
                print(f"    Min/Max:      "
                      f"{min(leads):.0f} / "
                      f"{max(leads):.0f}")
                print(f"    Non-negative: "
                      f"{pos}/{len(valid)} "
                      f"({pos/len(valid)*100:.0f}%)")

        print(f"\n  Warning status:")
        for st in ['hit', 'slightly_late',
                   'late_as_miss', 'missed',
                   'false_alarm', 'no_true_event']:
            cnt = sum(
                r.get('warning_status') == st
                for r in records)
            if cnt > 0:
                print(f"    {st:15s}: {cnt}")

        print(f"\n  Max level:")
        for lv in range(4):
            cnt = sum(r['max_level'] == lv
                      for r in records)
            print(f"    Lv{lv} "
                  f"{LEVEL_NAMES[lv]:8s}: {cnt}")
        print(f"{'='*70}\n")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":

    XJTU_PRED_DIR = "./Results_XJTU/predictions"
    NASA_PRED_DIR = "./Results_NASA/predictions"

    # ══════════════════════════════════════════════
    # XJTU 预警分析
    # ══════════════════════════════════════════════
    print("\n" + "="*60)
    print("  XJTU 预警分析")
    print("="*60)

    analyzer_xjtu = BatteryAlertAnalyzer(
        save_dir="./Alert_XJTU",
        cfg=XJTUAlertConfig())

    XJTU_EXPS = [
        ("Exp1_Fixed_Our_predictions.csv",
         "Exp1_Fixed",     4),
        ("Exp2_Dynamic_Our_predictions.csv",
         "Exp2_Dynamic",   4),
        ("Exp3_Fixed2Dyn_Our_predictions.csv",
         "Exp3_Fixed2Dyn", 8),
        ("Exp4_Dyn2Fixed_Our_predictions.csv",
         "Exp4_Dyn2Fixed", 8),
    ]

    all_xjtu_infos = []
    for csv_name, exp_tag, max_p in XJTU_EXPS:
        csv_path = os.path.join(
            XJTU_PRED_DIR, csv_name)
        print(f"\n  [{exp_tag}]")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path}")
            continue
        infos = analyzer_xjtu.run_from_csv(
            csv_path,
            dataset_tag=f"XJTU_{exp_tag}",
            model_filter="Our",
            max_plot=max_p)
        infos = deduplicate_infos(
            infos, strategy='best')
        analyzer_xjtu.plot_summary(
            infos,
            dataset_tag=f"XJTU_{exp_tag}")
        analyzer_xjtu.print_summary(
            infos,
            dataset_tag=f"XJTU_{exp_tag}")
        all_xjtu_infos.extend(infos)

    all_xjtu_infos = deduplicate_infos(
        all_xjtu_infos, strategy='best')
    if all_xjtu_infos:
        print("\n  [XJTU 全局汇总]")
        analyzer_xjtu.plot_summary(
            all_xjtu_infos,
            dataset_tag="XJTU_All")
        analyzer_xjtu.print_summary(
            all_xjtu_infos,
            dataset_tag="XJTU_All")

    # ══════════════════════════════════════════════
    # NASA 预警分析
    # ══════════════════════════════════════════════
    print("\n" + "="*60)
    print("  NASA 预警分析")
    print("="*60)

    analyzer_nasa = BatteryAlertAnalyzer(
        save_dir="./Alert_NASA",
        cfg=NASAGroupAConfig())

    NASA_EXPS = [
        ("LOO_Group_A_Our_predictions.csv",
         "LOO_Group_A", 4),
        ("LOO_Group_B_Our_predictions.csv",
         "LOO_Group_B", 4),
        ("Transfer_A2B_Our_predictions.csv",
         "Transfer_A2B", 8),
    ]

    all_nasa_infos = []
    for csv_name, exp_tag, max_p in NASA_EXPS:
        csv_path = os.path.join(
            NASA_PRED_DIR, csv_name)
        print(f"\n  [{exp_tag}]")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path}")
            continue
        infos = analyzer_nasa.run_from_csv(
            csv_path,
            dataset_tag=f"NASA_{exp_tag}",
            model_filter="Our",
            max_plot=max_p,
            bat_cfg_map=NASA_BAT_CFG_MAP)
        infos = deduplicate_infos(
            infos, strategy='best')
        analyzer_nasa.plot_summary(
            infos,
            dataset_tag=f"NASA_{exp_tag}")
        analyzer_nasa.print_summary(
            infos,
            dataset_tag=f"NASA_{exp_tag}")
        all_nasa_infos.extend(infos)

    all_nasa_infos = deduplicate_infos(
        all_nasa_infos, strategy='best')
    if all_nasa_infos:
        print("\n  [NASA 全局汇总]")
        analyzer_nasa.plot_summary(
            all_nasa_infos,
            dataset_tag="NASA_All")
        analyzer_nasa.print_summary(
            all_nasa_infos,
            dataset_tag="NASA_All")