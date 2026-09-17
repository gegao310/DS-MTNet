"""
XJTU电池统一特征提取 (双标签版本)
同时提取:
  输入特征: 统一 14 维循环级特征（严格对应论文 Table 1）
            └ 充电阶段 7 维: CC 4 维 (t_CC, E_CC, dVdt, H_V_CC)
                           + CV 3 维 (Q_CV, E_CV, H_I_CV)
            └ 放电阶段 7 维: Q_dis, E_dis, I_dis_mean,
                           V_dis_mean, V_dis_min, V_dis_std, k_V_dis
  标签1: Target_T_max  (温度预测)
  标签2: Target_SOH    (SOH预测, 为后续联合预测准备)

列名与顺序由 feature_schema.py 统一定义（与 NASA 脚本共用），
保证跨域迁移时两侧特征维度一致 (F = 14)。

修复: Dynamic域(Batch-3/4)加入放电倍率信息
      解决原版只有充电特征导致Dynamic域预测失效的问题

输出: unified_Batch-X.pkl (循环级, 双标签)
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew
import warnings
from tqdm import tqdm

from feature_schema import (
    CHARGE_FEATURES, DISCHARGE_FEATURES, FEATURE_NAMES,
    N_FEATURES, FEATURE_UNITS, check_feature_order,
)

warnings.filterwarnings("ignore")


class UnifiedBatteryExtractor:
    """
    统一特征提取器
    同时处理充电阶段和放电阶段
    输出双标签: T_max + SOH
    """

    def __init__(self, v_cc_start=3.9, v_cc_end=4.0):
        self.v_cc_start = v_cc_start
        self.v_cc_end   = v_cc_end

    # ==========================================
    # 工具函数
    # ==========================================
    def _calc_entropy(self, signal, bins=64):
        if len(signal) < 2 or np.all(signal == signal[0]):
            return 0.0
        sig_norm = (signal - np.min(signal)) / (
            np.max(signal) - np.min(signal) + 1e-8
        )
        counts, _ = np.histogram(
            sig_norm, bins=bins, density=True
        )
        p = counts[counts > 0]
        return float(-np.sum(p * np.log2(p + 1e-8)))

    def _safe_skew(self, arr):
        if len(arr) < 3:
            return 0.0
        return float(skew(arr, nan_policy='omit'))

    def _extract_charge_features(self, time, v, i, t):
        """
        充电阶段特征提取
        输入: 充电段的 time/v/i/t
        输出: dict, 固定 7 维 (CC 4 维 + CV 3 维)，对应论文 Table 1

        单位约定: 原始 relative_time_min 单位为 min，
                  积分时除以 60 转为 h，故
                  ∫V·I dt → Wh, ∫I dt → Ah
        """
        feats = {}

        if len(time) < 2:
            # 数据不足：返回全零占位
            return {k: 0.0 for k in CHARGE_FEATURES}

        dt      = (time - time[0]) / 60.0   # min -> h
        power   = v * i
        cv_start  = np.where(v >= 4.199)[0]
        split_idx = cv_start[0] if len(cv_start) > 0 \
                    else np.argmax(v)
        split_idx = max(1, min(split_idx, len(v) - 2))

        time_cc = time[:split_idx]
        v_cc    = v[:split_idx]
        i_cc    = i[:split_idx]
        time_cv = time[split_idx:]
        v_cv    = v[split_idx:]
        i_cv    = i[split_idx:]

        # CC段特征 (4 维)
        if len(time_cc) > 1:
            dt_cc = (time_cc - time_cc[0]) / 60.0      # min -> h
            feats['E_CC'] = float(
                np.trapz(v_cc * i_cc, dt_cc)           # Wh
            )
            feats['t_CC'] = float(
                time_cc[-1] - time_cc[0]               # min
            )
            feats['H_V_CC'] = self._calc_entropy(v_cc)

            # dV/dt (极化特征)  [V/min]
            idx_dv = np.where(
                (v_cc >= self.v_cc_start) &
                (v_cc <= self.v_cc_end)
            )[0]
            if len(idx_dv) > 5:
                dt_seg = (
                    time_cc[idx_dv[-1]] -
                    time_cc[idx_dv[0]]
                )
                feats['dVdt'] = float(
                    (v_cc[idx_dv[-1]] - v_cc[idx_dv[0]]) /
                    (dt_seg + 1e-8)
                )
            else:
                feats['dVdt'] = 0.0
        else:
            feats.update({
                'E_CC': 0.0, 't_CC': 0.0,
                'H_V_CC': 0.0, 'dVdt': 0.0
            })

        # CV段特征 (3 维)
        if len(time_cv) > 5:
            dt_cv = (time_cv - time_cv[0]) / 60.0      # min -> h
            feats['E_CV'] = float(
                np.trapz(v_cv * i_cv, dt_cv)           # Wh
            )
            feats['Q_CV']    = float(
                np.trapz(i_cv, dt_cv)                  # Ah
            )
            feats['H_I_CV'] = self._calc_entropy(i_cv)
        else:
            feats.update({
                'E_CV': 0.0,
                'Q_CV': 0.0,
                'H_I_CV': 0.0
            })

        # 保证 7 个键齐全且顺序固定
        return {k: float(feats.get(k, 0.0))
                for k in CHARGE_FEATURES}

    def _extract_discharge_features(self, time, v, i, t):
        """
        放电阶段特征提取
        输入: 放电段的 time/v/i/t (i为负值)
        输出: dict, 固定 7 维，对应论文 Table 1 的放电阶段

        关键: 放电倍率信息在这里
        → 修复Dynamic域预测失效的核心
        """
        feats = {}

        # 取绝对值方便计算
        i_abs = np.abs(i)

        if len(time) < 2:
            return {k: 0.0 for k in DISCHARGE_FEATURES}

        dt    = (time - time[0]) / 60.0     # min -> h
        power = v * i_abs

        # 容量 [Ah] 与能量 [Wh]
        feats['Q_dis'] = float(np.trapz(i_abs, dt))
        feats['E_dis'] = float(np.trapz(power,  dt))

        # 电流统计 (反映放电倍率) [A]
        feats['I_dis_mean'] = float(np.mean(i_abs))

        # 电压统计 (反映内阻和极化) [V]
        feats['V_dis_mean'] = float(np.mean(v))
        feats['V_dis_min']  = float(np.min(v))
        feats['V_dis_std']  = float(np.std(v))

        # 电压下降斜率 (反映老化程度) [V/step]
        if len(v) > 5:
            x = np.linspace(0, 1, len(v))
            feats['k_V_dis'] = float(
                np.polyfit(x, v, 1)[0]
            )
        else:
            feats['k_V_dis'] = 0.0

        # 保证 7 个键齐全且顺序固定
        return {k: float(feats.get(k, 0.0))
                for k in DISCHARGE_FEATURES}

    def _extract_temp_labels(self, t_charge, t_discharge):
        """
        温度标签提取
        取充放电全过程的温度统计
        """
        t_all = np.concatenate([t_charge, t_discharge])

        if len(t_all) < 1:
            return {
                'Target_T_max':  np.nan,
                'Target_T_mean': np.nan,
                'Target_T_rise': np.nan
            }

        return {
            'Target_T_max':  float(np.max(t_all)),
            'Target_T_mean': float(np.mean(t_all)),
            'Target_T_rise': float(
                np.max(t_all) - t_all[0]
            )
        }

    def extract_from_cycle(self, cycle_data):
        """
        从单个cycle提取全部特征和标签

        返回: dict 或 None (数据不足时)
        包含:
          充电特征 (7维: CC 4 + CV 3)
          放电特征 (7维)
          温度标签 (3个)
          放电容量 (用于SOH计算)
        """
        try:
            time = cycle_data['relative_time_min'].flatten()
            v    = cycle_data['voltage_V'].flatten()
            i    = cycle_data['current_A'].flatten()
            t    = cycle_data['temperature_C'].flatten()
        except Exception:
            return None

        if len(time) < 20:
            return None

        # 分离充电段和放电段
        charge_idx    = i > 0.01    # 充电: 电流为正
        discharge_idx = i < -0.01   # 放电: 电流为负

        n_charge    = np.sum(charge_idx)
        n_discharge = np.sum(discharge_idx)

        # 充电数据不足则跳过
        if n_charge < 10:
            return None

        # 提取各段数据
        time_c = time[charge_idx]
        v_c    = v[charge_idx]
        i_c    = i[charge_idx]
        t_c    = t[charge_idx]

        time_d = time[discharge_idx]
        v_d    = v[discharge_idx]
        i_d    = i[discharge_idx]
        t_d    = t[discharge_idx]

        # 组合所有特征
        result = {}

        # 1. 充电特征 (原有逻辑)
        charge_feats = self._extract_charge_features(
            time_c, v_c, i_c, t_c
        )
        result.update(charge_feats)

        # 2. 放电特征 (新增)
        if n_discharge >= 10:
            discharge_feats = self._extract_discharge_features(
                time_d, v_d, i_d, t_d
            )
        else:
            # 放电数据不足: 用零填充
            discharge_feats = self._extract_discharge_features(
                np.array([0, 1]),
                np.array([3.0, 2.5]),
                np.array([-1.0, -1.0]),
                np.array([25.0, 25.0])
            )
            discharge_feats = {k: 0.0
                               for k in discharge_feats}
        result.update(discharge_feats)

        # 3. 温度标签
        temp_labels = self._extract_temp_labels(t_c, t_d)
        result.update(temp_labels)

        # 4. 放电容量 (SOH计算用, 单独保存)
        result['_raw_Q_dis'] = result['Q_dis']

        return result


# ==========================================
# 批次处理
# ==========================================
def process_batch(batch_root, batch_name,
                  extractor, output_dir="extracted_features"):
    """
    处理一个Batch的全部mat文件
    计算SOH并保存统一pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    mat_files = sorted(
        [f for f in os.listdir(batch_root)
         if f.endswith('.mat')]
    )

    if not mat_files:
        print(f"  [WARN] {batch_name} 无mat文件")
        return None

    all_records = []
    print(f"\n{'='*55}")
    print(f"  处理 {batch_name} ({len(mat_files)}个文件)")
    print(f"{'='*55}")

    for mat_file in tqdm(mat_files,
                         desc=f"  {batch_name}"):
        try:
            path = os.path.join(batch_root, mat_file)
            data = loadmat(path, simplify_cells=True)

            # 兼容不同mat结构
            if 'data' in data:
                cycles = data['data']
            elif 'battery' in data:
                cycles = data['battery'].get('data', None)
            else:
                # 尝试找第一个非系统键
                keys = [k for k in data.keys()
                        if not k.startswith('_')]
                cycles = data[keys[0]] if keys else None

            if cycles is None:
                continue

            # 电池ID
            battery_id = (
                mat_file.replace('.mat', '')
                        .replace('-', '_')
            )
            full_id = f"{batch_name}_{battery_id}"

            # 逐循环提取
            battery_records = []
            for cycle_idx in range(len(cycles)):
                try:
                    cycle = cycles[cycle_idx]
                    res   = extractor.extract_from_cycle(
                        cycle
                    )
                    if res is None:
                        continue
                    res['battery_id']  = full_id
                    res['cycle_count'] = cycle_idx + 1
                    battery_records.append(res)
                except Exception:
                    continue

            if len(battery_records) < SEQ_LEN_MIN:
                print(f"\n  [SKIP] {full_id}: "
                      f"只有{len(battery_records)}个有效循环")
                continue

            # ── SOH计算 ──────────────────────────────
            # 用第一个有效循环的放电容量作为额定容量
            q_values = np.array(
                [r['_raw_Q_dis'] for r in battery_records]
            )

            # 找第一个非零放电容量
            valid_q = q_values[q_values > 0.01]
            if len(valid_q) == 0:
                print(f"\n  [SKIP] {full_id}: "
                      f"无有效放电容量")
                continue

            q_nominal = valid_q[0]  # 第一个有效循环的容量

            for r in battery_records:
                q = r['_raw_Q_dis']
                if q > 0.01:
                    r['Target_SOH'] = float(
                        min(q / q_nominal, 1.05)
                    )
                else:
                    # 放电容量为0 (第一个容量测量循环)
                    # 用1.0填充
                    r['Target_SOH'] = 1.0

                # 清理临时字段
                del r['_raw_Q_dis']

            all_records.extend(battery_records)

            print(f"\n  ✅ {full_id}: "
                  f"{len(battery_records)}个循环, "
                  f"Q_nominal={q_nominal:.4f}Ah, "
                  f"SOH范围=[{min(r['Target_SOH'] for r in battery_records):.3f}, "
                  f"{max(r['Target_SOH'] for r in battery_records):.3f}]")

        except Exception as e:
            print(f"\n  [ERROR] {mat_file}: {e}")
            continue

    if not all_records:
        print(f"  [WARN] {batch_name} 无有效数据")
        return None

    df = pd.DataFrame(all_records)

    # 列排序: ID列 → cycle列 → 特征列 → 标签列
    id_cols    = ['battery_id', 'cycle_count']
    label_cols = ['Target_T_max', 'Target_T_mean',
                  'Target_T_rise', 'Target_SOH']
    raw_feats  = [c for c in df.columns
                  if c not in id_cols + label_cols]
    # 按论文 Table 1 的 14 维顺序对齐，缺列/多列会直接报错
    feat_cols  = check_feature_order(raw_feats)

    df = df[id_cols + feat_cols + label_cols]

    # 处理NaN
    df[feat_cols]  = df[feat_cols].fillna(0.0)
    df[label_cols] = df[label_cols].ffill().bfill()

    # 保存
    save_path = os.path.join(
        output_dir, f"unified_{batch_name}.pkl"
    )
    df.to_pickle(save_path)

    # 打印统计
    n_bats = df['battery_id'].nunique()
    print(f"\n  {batch_name} 汇总:")
    print(f"    电池数:    {n_bats}")
    print(f"    总循环数:  {len(df)}")
    print(f"    特征维度:  {len(feat_cols)} "
          f"(须等于统一特征表 {N_FEATURES} 维)")
    for c in feat_cols:
        print(f"      - {c:12s} [{FEATURE_UNITS.get(c, '-')}]")
    print(f"    T_max范围: "
          f"[{df['Target_T_max'].min():.1f}, "
          f"{df['Target_T_max'].max():.1f}]℃")
    print(f"    SOH范围:   "
          f"[{df['Target_SOH'].min():.3f}, "
          f"{df['Target_SOH'].max():.3f}]")
    print(f"    已保存 → {save_path}")

    return df


# ==========================================
# 验证函数: 检查提取结果是否合理
# ==========================================
def validate_extraction(df, batch_name):
    """
    对提取结果做基本检验
    打印每块电池的关键统计
    """
    print(f"\n{'─'*55}")
    print(f"  验证: {batch_name}")
    print(f"{'─'*55}")

    for bat_id, group in df.groupby('battery_id'):
        soh_start = group['Target_SOH'].iloc[0]
        soh_end   = group['Target_SOH'].iloc[-1]
        t_mean    = group['Target_T_max'].mean()
        dis_i     = group['I_dis_mean'].mean()
        n_cycles  = len(group)

        print(f"  {bat_id:35s} "
              f"循环={n_cycles:4d}  "
              f"SOH: {soh_start:.3f}→{soh_end:.3f}  "
              f"T_max均值={t_mean:.1f}℃  "
              f"放电I均值={dis_i:.3f}A")

    # 检查Dynamic域的放电电流变化
    if 'Batch-3' in batch_name or 'Batch-4' in batch_name:
        print(f"\n  Dynamic域放电电流分布:")
        print(f"    I_dis_mean: "
              f"[{df['I_dis_mean'].min():.3f}, "
              f"{df['I_dis_mean'].max():.3f}] A")
        print(f"    V_dis_min:  "
              f"[{df['V_dis_min'].min():.3f}, "
              f"{df['V_dis_min'].max():.3f}] V")
        print(f"  → 如果范围够宽(如0.5~5A), 说明放电特征提取成功")


# ==========================================
# 主函数
# ==========================================
SEQ_LEN_MIN = 25  # 电池最少有效循环数

# ── 路径配置（开源版：请按需修改为你的本地路径）──
# base_path: XJTU 数据集根目录（内含 Batch-1/2/3/4 子目录，每个子目录下是 .mat 文件）
# output_dir: 提取后的特征输出目录
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="XJTU 电池循环级特征提取")
    parser.add_argument(
        "--base_path", type=str, default="./XJTU_dataset",
        help="XJTU 数据集根目录（默认 ./XJTU_dataset）")
    parser.add_argument(
        "--output_dir", type=str, default="./extracted_features",
        help="特征输出目录（默认 ./extracted_features）")
    args = parser.parse_args()

    base_path  = args.base_path
    output_dir = args.output_dir

    extractor = UnifiedBatteryExtractor()

    batches = [f"Batch-{i}" for i in range(1, 5)]
    all_dfs = {}

    for b_name in batches:
        b_dir = os.path.join(base_path, b_name)
        if not os.path.exists(b_dir):
            print(f"  [WARN] 路径不存在: {b_dir}")
            continue

        df = process_batch(
            b_dir, b_name, extractor, output_dir
        )
        if df is not None:
            all_dfs[b_name] = df
            validate_extraction(df, b_name)

    # 全局汇总
    print(f"\n{'='*55}")
    print(f"  全部完成!")
    print(f"{'='*55}")

    for bname, df in all_dfs.items():
        feat_cols = [
            c for c in df.columns
            if c not in ['battery_id', 'cycle_count',
                         'Target_T_max', 'Target_T_mean',
                         'Target_T_rise', 'Target_SOH']
        ]
        print(f"  {bname}: "
              f"{df['battery_id'].nunique()}只电池, "
              f"{len(df)}个循环, "
              f"{len(feat_cols)}维特征")

    print(f"\n  输出文件:")
    for b_name in batches:
        fpath = os.path.join(
            output_dir, f"unified_{b_name}.pkl"
        )
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"    unified_{b_name}.pkl  "
                  f"{size:.1f} KB")

    print(f"\n  统一特征列说明 "
          f"(共 {N_FEATURES} 维, 与论文 Table 1 一致):")
    print(f"    CC 充电(4维): t_CC / E_CC / dVdt / H_V_CC")
    print(f"    CV 充电(3维): Q_CV / E_CV / H_I_CV")
    print(f"    放电(7维)  : Q_dis / E_dis / I_dis_mean / "
          f"V_dis_mean / V_dis_min / V_dis_std / k_V_dis")
    print(f"    温度标签    : Target_T_max (主要使用)")
    print(f"    SOH标签     : Target_SOH (为后续联合预测准备)")


if __name__ == "__main__":
    main()