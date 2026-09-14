import os
import re
import warnings
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import linregress
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================
# 开源版：请按需修改为你本地 NASA 数据集的路径，或通过命令行参数覆盖
RAW_DATA_DIR = "./NASA_dataset"     # NASA 原始 .mat 文件目录
OUT_FEATURE_DIR = "./nasa_features"  # 提取后的特征输出目录

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
    if len(df) < MIN_POINTS_PER_CYCLE:
        return None

    t = df["time_s"].values.astype(np.float64)
    v = df["voltage_V"].values.astype(np.float64)
    i = df["current_A"].values.astype(np.float64)
    temp = df["temperature_C"].values.astype(np.float64)
    soc = df["soc"].values.astype(np.float64)
    cap = float(df["capacity_Ah"].iloc[0])

    t_rel = t - t[0]
    duration = float(t_rel[-1]) if len(t_rel) > 0 else 0.0
    duration = max(duration, 1e-6)

    feats = {}

    # 基本信息
    feats["cycle_num"] = cycle_num
    feats["cycle_index"] = int(df["cycle_index"].iloc[0])
    feats["capacity_Ah"] = cap
    feats["discharge_duration"] = duration

    # 电压特征
    feats["v_start"] = float(v[0])
    feats["v_end"] = float(v[-1])
    feats["v_mean"] = float(np.mean(v))
    feats["v_std"] = float(np.std(v))
    feats["v_min"] = float(np.min(v))
    feats["v_max"] = float(np.max(v))
    feats["v_range"] = float(v[0] - v[-1])

    n = len(v)
    n30 = max(int(n * 0.3), 2)
    feats["v_slope_early"] = safe_linregress(t_rel[:n30], v[:n30])
    feats["v_slope_late"] = safe_linregress(t_rel[-n30:], v[-n30:])

    feats["v_at_soc80"] = interp_y_at_x(soc, v, 0.8)
    feats["v_at_soc50"] = interp_y_at_x(soc, v, 0.5)
    feats["v_at_soc20"] = interp_y_at_x(soc, v, 0.2)

    # 电流特征
    feats["i_mean_abs"] = float(np.mean(np.abs(i)))
    feats["i_std"] = float(np.std(i))
    feats["i_min"] = float(np.min(i))
    feats["i_max"] = float(np.max(i))

    # 温度特征
    feats["t_start"] = float(temp[0])
    feats["t_end"] = float(temp[-1])
    feats["t_max"] = float(np.max(temp))
    feats["t_mean"] = float(np.mean(temp))
    feats["t_std"] = float(np.std(temp))
    feats["t_rise"] = float(np.max(temp) - temp[0])

    # SOC特征
    feats["soc_start"] = float(soc[0])
    feats["soc_end"] = float(soc[-1])
    feats["soc_range"] = float(soc[0] - soc[-1])
    feats["time_to_soc50"] = interp_y_at_x(soc, t_rel, 0.5)

    # 能量 / 功率近似
    dt = np.diff(t_rel, prepend=0.0)
    dt = np.clip(dt, 0, None)
    energy_ws = np.sum(v * np.abs(i) * dt)
    feats["energy_Wh"] = float(energy_ws / 3600.0)
    feats["avg_power_W"] = float(energy_ws / duration)

    # 简单内阻 proxy
    feats["v_drop_per_A"] = float((v[0] - v[-1]) / (np.mean(np.abs(i)) + 1e-6))

    # 老化目标
    feats["Target_SOH"] = float(cap / ref_capacity)
    feats["Target_T_max"] = float(np.max(temp))
    feats["Target_T_mean"] = float(np.mean(temp))
    feats["Target_T_rise"] = float(np.max(temp) - temp[0])

    # 对标你原始代码字段
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

    feat_df = pd.DataFrame(rows).sort_values("cycle_num").reset_index(drop=True)

    # 衍生特征
    feat_df["capacity_fade_rate"] = feat_df["capacity_Ah"].diff().fillna(0.0)
    feat_df["capacity_fade_pct"] = feat_df["capacity_Ah"] / ref_capacity
    feat_df["cycle_ratio"] = np.arange(len(feat_df)) / max(len(feat_df) - 1, 1)

    for col in ["capacity_Ah", "t_max", "t_mean", "v_mean", "energy_Wh"]:
        feat_df[f"prev_{col}"] = feat_df[col].shift(1).bfill()
        feat_df[f"delta_{col}"] = feat_df[col].diff().fillna(0.0)

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

    # 打印特征列1
    sample = next(iter(all_dfs.values()))
    exclude_cols = {
        "battery_id", "cycle_count", "cycle_num", "cycle_index",
        "Target_SOH", "Target_T_max", "Target_T_mean", "Target_T_rise"
    }
    feat_cols = [c for c in sample.columns if c not in exclude_cols]

    print(f"\nFeature columns ({len(feat_cols)} dims):")
    for c in feat_cols:
        print(" ", c)

    # 保存一个总览表
    overview = []
    for bat, df in all_dfs.items():
        overview.append({
            "battery": bat,
            "n_cycles": len(df),
            "soh_start": float(df["Target_SOH"].iloc[0]),
            "soh_end": float(df["Target_SOH"].iloc[-1]),
            "cap_start": float(df["capacity_Ah"].iloc[0]),
            "cap_end": float(df["capacity_Ah"].iloc[-1]),
        })
    overview_df = pd.DataFrame(overview)
    overview_csv = os.path.join(out_dir, "battery_overview.csv")
    overview_df.to_csv(overview_csv, index=False)
    print(f"\nSaved overview: {overview_csv}")

    return all_dfs


# ============================================================
if __name__ == "__main__":
    all_dfs = preprocess_all()
    print("\nDone.")