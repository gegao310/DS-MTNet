import os
import re
import warnings
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import linregress
from scipy.interpolate import interp1d

from feature_schema import (
    CHARGE_FEATURES, DISCHARGE_FEATURES, FEATURE_NAMES,
    N_FEATURES, FEATURE_UNITS, empty_charge_features,
    check_feature_order,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================
# 开源版：请按需修改为你本地 NASA 数据集的路径，或通过命令行参数覆盖
RAW_DATA_DIR = "./NASA_dataset"     # NASA 原始 .mat 文件目录
OUT_FEATURE_DIR = "./nasa_features"  # 提取后的特征输出目录

# ── 统一 14 维特征表（与 XJTU_processing.py / 论文 Table 1 共用）──
# NASA 数据集只有放电循环记录，没有 CC / CV 充电阶段的原始数据，
# 因此 7 个充电阶段特征按论文所述用 0 填充（zero-padding），
# 以保证与 XJTU 侧的特征维度一致（F = 14），支持跨域权重迁移。

# 先用这4个经典电池做验证，和你前面的设计保持一致
BATTERIES = BATTERIES = [
    "B0005", "B0006", "B0007", "B0018",
    "B0045", "B0046", "B0047", "B0048",
    "B0053", "B0054", "B0055", "B0056",
]

# 如果你想自动处理目录下所有 Bxxxx.mat，可改成：
# BATTERIES = None

CURRENT_ACTIVE_THRESHOLD = 1.0   # |current| > 1A 视为有效放电
MIN_POINTS_PER_CYCLE = 10


# ============================================================
# 1. 自动发现电池
# ============================================================
def discover_batteries(data_dir):
    bats = []
    for fn in os.listdir(data_dir):
        if re.fullmatch(r"B\d{4}\.mat", fn):
            bats.append(fn[:-4])
    return sorted(bats)


# ============================================================
# 2. 从 .mat 提取 discharge cycles
# ============================================================
def extract_discharge_cycles_from_mat(mat_path, battery_name):
    """
    从 NASA 原始 .mat 中提取所有 discharge 循环
    返回: list[pd.DataFrame]
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if battery_name not in mat:
        raise KeyError(f"{battery_name} not found in {mat_path}")

    battery = mat[battery_name]
    cycles = np.atleast_1d(battery.cycle)

    discharge_cycles = []

    for i, cycle in enumerate(cycles):
        try:
            cycle_type = str(cycle.type).strip().lower()
        except Exception:
            continue

        if cycle_type != "discharge":
            continue

        try:
            data = cycle.data

            time = np.asarray(np.atleast_1d(data.Time), dtype=np.float64).reshape(-1)
            voltage = np.asarray(np.atleast_1d(data.Voltage_measured), dtype=np.float64).reshape(-1)
            current = np.asarray(np.atleast_1d(data.Current_measured), dtype=np.float64).reshape(-1)
            temperature = np.asarray(np.atleast_1d(data.Temperature_measured), dtype=np.float64).reshape(-1)

            capacity = float(np.asarray(data.Capacity).squeeze())
        except Exception as e:
            print(f"  [WARN] skip {battery_name} cycle {i}: parse failed -> {e}")
            continue

        # 对齐长度
        L = min(len(time), len(voltage), len(current), len(temperature))
        if L < MIN_POINTS_PER_CYCLE:
            continue

        time = time[:L]
        voltage = voltage[:L]
        current = current[:L]
        temperature = temperature[:L]

        if (not np.isfinite(capacity)) or capacity <= 0:
            continue

        # 构造 SOC: 用该循环实际容量做 Coulomb counting
        dt = np.diff(time, prepend=time[0])
        dt = np.clip(dt, 0, None)

        charge_removed = np.cumsum(np.abs(current) * dt)
        soc = 1.0 - charge_removed / (capacity * 3600.0)
        soc = np.clip(soc, 0.0, 1.0)

        df = pd.DataFrame({
            "time_s": time,
            "voltage_V": voltage,
            "current_A": current,
            "temperature_C": temperature,
            "soc": soc,
            "capacity_Ah": capacity,
            "cycle_index": i,
            "battery": battery_name,
        })

        discharge_cycles.append(df)

    return discharge_cycles


# ============================================================
# 3. 清洗单循环
# ============================================================
def clean_cycle(df):
    df = df.copy()

    keep = np.isfinite(df["time_s"]) & np.isfinite(df["voltage_V"]) \
           & np.isfinite(df["current_A"]) & np.isfinite(df["temperature_C"])
    df = df[keep].copy()

    df = df.sort_values("time_s").drop_duplicates(subset="time_s").reset_index(drop=True)

    abs_i = np.abs(df["current_A"].values)
    nz = abs_i[abs_i > 0.02]
    if len(nz) == 0:
        return df.iloc[:0].copy()

    # 自适应活动电流阈值
    active_level = np.median(nz)
    thr = max(0.05, 0.2 * active_level)

    df = df[abs_i > thr].reset_index(drop=True)
    return df


# ============================================================
# 4. 小工具
# ============================================================
def safe_linregress(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0.0
    try:
        slope, _, _, _, _ = linregress(x, y)
        if np.isfinite(slope):
            return float(slope)
    except Exception:
        pass
    return 0.0


def interp_y_at_x(x, y, xq):
    """
    给定散点 (x,y)，插值求 y(xq)
    自动排序、去重
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < 2:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    if len(x_unique) < 2:
        return np.nan

    try:
        f = interp1d(x_unique, y_unique, bounds_error=False, fill_value="extrapolate")
        val = float(f(xq))
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


# ============================================================
# 5. 单循环 -> 循环级特征
# ============================================================
def extract_cycle_features(df, cycle_num, ref_capacity):
    """
    单放电循环 -> 统一 14 维循环级特征

    与 XJTU_processing.py / 论文 Table 1 完全同名、同序：
      - 充电阶段 7 维 (CC 4 + CV 3): NASA 无对应工况记录 -> 零填充
      - 放电阶段 7 维: 由放电段的 V / I / t 积分与统计得到

    单位约定: time_s 为秒，积分时除以 3600 转为 h，
             故 ∫V·|I| dt -> Wh, ∫|I| dt -> Ah
    """
    if len(df) < MIN_POINTS_PER_CYCLE:
        return None

    t    = df["time_s"].values.astype(np.float64)
    v    = df["voltage_V"].values.astype(np.float64)
    i    = df["current_A"].values.astype(np.float64)
    temp = df["temperature_C"].values.astype(np.float64)
    cap  = float(df["capacity_Ah"].iloc[0])

    i_abs = np.abs(i)
    dt_h  = (t - t[0]) / 3600.0            # s -> h
    dt_h  = np.clip(dt_h, 0.0, None)

    feats = {}

    # ── 充电阶段 7 维: 无记录, 零填充 ────────────────────────
    feats.update(empty_charge_features())

    # ── 放电阶段 7 维 ────────────────────────────────────────
    feats["Q_dis"]      = float(np.trapz(i_abs, dt_h))          # Ah
    feats["E_dis"]      = float(np.trapz(v * i_abs, dt_h))      # Wh
    feats["I_dis_mean"] = float(np.mean(i_abs))                 # A
    feats["V_dis_mean"] = float(np.mean(v))                     # V
    feats["V_dis_min"]  = float(np.min(v))                      # V
    feats["V_dis_std"]  = float(np.std(v))                      # V

    # 放电电压衰减斜率 [V/step]: 对归一化时间轴做一阶线性拟合
    if len(v) > 5:
        x = np.linspace(0.0, 1.0, len(v))
        feats["k_V_dis"] = float(np.polyfit(x, v, 1)[0])
    else:
        feats["k_V_dis"] = 0.0

    # ── 老化 / 温度目标 ──────────────────────────────────────
    feats["Target_SOH"]    = float(cap / ref_capacity)
    feats["Target_T_max"]  = float(np.max(temp))
    feats["Target_T_mean"] = float(np.mean(temp))
    feats["Target_T_rise"] = float(np.max(temp) - temp[0])

    feats["cycle_count"] = cycle_num

    return feats


# ============================================================
# 6. 构造单电池特征表
# ============================================================
def build_battery_feature_table(battery_name, raw_dir):
    mat_path = os.path.join(raw_dir, f"{battery_name}.mat")
    if not os.path.isfile(mat_path):
        print(f"  [WARN] {mat_path} not found")
        return None

    cycles = extract_discharge_cycles_from_mat(mat_path, battery_name)
    print(f"  {battery_name}: {len(cycles)} discharge cycles found in MAT")

    cleaned_cycles = []
    for df in cycles:
        cdf = clean_cycle(df)
        if len(cdf) >= MIN_POINTS_PER_CYCLE:
            cleaned_cycles.append(cdf)

    if len(cleaned_cycles) == 0:
        print(f"  [WARN] {battery_name}: no valid cycles after cleaning")
        return None

    ref_capacity = float(cleaned_cycles[0]["capacity_Ah"].iloc[0])

    rows = []
    for ci, df in enumerate(cleaned_cycles):
        feats = extract_cycle_features(df, cycle_num=ci, ref_capacity=ref_capacity)
        if feats is None:
            continue
        feats["battery_id"] = battery_name
        rows.append(feats)

    if len(rows) == 0:
        print(f"  [WARN] {battery_name}: no feature rows extracted")
        return None

    feat_df = pd.DataFrame(rows).sort_values("cycle_count").reset_index(drop=True)

    # ── 统一特征表: 严格保留 14 维并按 Table 1 顺序排列 ──────
    target_cols = ["Target_SOH", "Target_T_max",
                   "Target_T_mean", "Target_T_rise"]
    ordered = (["battery_id", "cycle_count"]
               + check_feature_order(
                   [c for c in feat_df.columns
                    if c not in ["battery_id", "cycle_count"] + target_cols])
               + target_cols)
    feat_df = feat_df[ordered]

    # 数值清洗
    num_cols = feat_df.select_dtypes(include=[np.number]).columns
    feat_df[num_cols] = feat_df[num_cols].replace([np.inf, -np.inf], np.nan)
    feat_df[num_cols] = feat_df[num_cols].fillna(feat_df[num_cols].median())
    feat_df[num_cols] = feat_df[num_cols].fillna(0.0)

    print(
        f"  {battery_name}: valid_cycles={len(feat_df)}, "
        f"SOH range=[{feat_df['Target_SOH'].min():.4f}, {feat_df['Target_SOH'].max():.4f}]"
    )

    return feat_df


# ============================================================
# 7. 处理全部电池
# ============================================================
def preprocess_all(raw_dir=RAW_DATA_DIR, out_dir=OUT_FEATURE_DIR, batteries=BATTERIES):
    os.makedirs(out_dir, exist_ok=True)

    if batteries is None:
        batteries = discover_batteries(raw_dir)

    if len(batteries) == 0:
        raise RuntimeError(f"No battery .mat files found in: {raw_dir}")

    print(f"Raw dir: {raw_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Batteries: {batteries}")

    all_dfs = {}

    for bat in batteries:
        print(f"\nProcessing {bat}...")
        df = build_battery_feature_table(bat, raw_dir)
        if df is None:
            continue

        save_path = os.path.join(out_dir, f"{bat}_features.pkl")
        df.to_pickle(save_path)
        print(f"  Saved: {save_path}")
        all_dfs[bat] = df

    if len(all_dfs) == 0:
        raise FileNotFoundError(
            "No battery feature tables were created.\n"
            "请检查：\n"
            "1) RAW_DATA_DIR 是否指向原始 .mat 目录；\n"
            "2) 目录下是否包含 B0005.mat 之类文件；\n"
            "3) scipy.io.loadmat 能否正常读取这些文件。"
        )

    # 统一特征列校验与打印
    sample = next(iter(all_dfs.values()))
    target_cols = {"Target_SOH", "Target_T_max",
                   "Target_T_mean", "Target_T_rise"}
    feat_cols = check_feature_order(
        [c for c in sample.columns
         if c not in {"battery_id", "cycle_count"} | target_cols]
    )

    print(f"\nFeature columns ({len(feat_cols)} dims, "
          f"must equal {N_FEATURES}):")
    for c in feat_cols:
        tag = "zero-padded" if c in CHARGE_FEATURES else "measured"
        print(f"  {c:12s} [{FEATURE_UNITS.get(c, '-'):6s}]  <- {tag}")

    # 保存一个总览表
    overview = []
    for bat, df in all_dfs.items():
        row = {
            "battery": bat,
            "n_cycles": len(df),
            "soh_start": float(df["Target_SOH"].iloc[0]),
            "soh_end": float(df["Target_SOH"].iloc[-1]),
        }
        for c in ("I_dis_mean", "V_dis_mean", "Q_dis", "E_dis"):
            row[c] = float(df[c].mean()) if c in df.columns else np.nan
        overview.append(row)
    overview_df = pd.DataFrame(overview)
    overview_csv = os.path.join(out_dir, "battery_overview.csv")
    overview_df.to_csv(overview_csv, index=False)
    print(f"\nSaved overview: {overview_csv}")

    return all_dfs


# ============================================================
if __name__ == "__main__":
    all_dfs = preprocess_all()
    print("\nDone.")