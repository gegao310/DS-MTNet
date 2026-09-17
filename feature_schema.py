"""
统一 14 维循环级特征表（对应论文 Table 1）

为保证 XJTU 与 NASA 两个数据集的特征向量维度一致（F = 14），
便于跨域权重迁移，所有特征提取脚本必须严格按本文件定义的
**列名** 与 **顺序** 输出特征。

- XJTU：CC / CV / 放电三个阶段的特征均可实测得到。
- NASA：仅有放电循环，CC / CV 阶段的 7 维特征无法获取，
        按论文所述用 0 填充（zero-padding），以保持统一维度。
"""

# ── 充电阶段：CC 4 维（论文单位）──────────────────────────
CC_FEATURES = [
    "t_CC",      # 恒流阶段时长              [min]
    "E_CC",      # 恒流阶段输入能量          [Wh]
    "dVdt",      # 电压变化率（极化）        [V/min]
    "H_V_CC",    # 恒流段电压香农熵          [bits]
]

# ── 充电阶段：CV 3 维 ──────────────────────────────────────
CV_FEATURES = [
    "Q_CV",      # 恒压阶段输入容量          [Ah]
    "E_CV",      # 恒压阶段输入能量          [Wh]
    "H_I_CV",    # 恒压段电流香农熵          [bits]
]

# ── 放电阶段：7 维 ─────────────────────────────────────────
DISCHARGE_FEATURES = [
    "Q_dis",        # 放电容量                [Ah]
    "E_dis",        # 放电能量                [Wh]
    "I_dis_mean",   # 平均放电电流            [A]
    "V_dis_mean",   # 平均放电电压            [V]
    "V_dis_min",    # 最低放电电压            [V]
    "V_dis_std",    # 放电电压标准差          [V]
    "k_V_dis",      # 放电电压衰减斜率        [V/step]
]

# ── 充电阶段全流程（CC + CV），共 7 维 ─────────────────────
CHARGE_FEATURES = CC_FEATURES + CV_FEATURES

# ── 统一特征向量顺序：7 充电 + 7 放电 = 14 维 ──────────────
FEATURE_NAMES = CHARGE_FEATURES + DISCHARGE_FEATURES

N_FEATURES = len(FEATURE_NAMES)   # = 14

FEATURE_UNITS = {
    "t_CC": "min", "E_CC": "Wh", "dVdt": "V/min", "H_V_CC": "bits",
    "Q_CV": "Ah", "E_CV": "Wh", "H_I_CV": "bits",
    "Q_dis": "Ah", "E_dis": "Wh", "I_dis_mean": "A",
    "V_dis_mean": "V", "V_dis_min": "V", "V_dis_std": "V",
    "k_V_dis": "V/step",
}


def empty_charge_features():
    """充电特征缺失时（如 NASA）的零填充占位"""
    return {k: 0.0 for k in CHARGE_FEATURES}


def check_feature_order(feat_cols):
    """
    校验 DataFrame 的特征列与统一 14 维表是否完全一致
    返回排序后的特征列；不一致时抛出 ValueError
    """
    missing = [c for c in FEATURE_NAMES if c not in feat_cols]
    extra = [c for c in feat_cols if c not in FEATURE_NAMES]
    if missing or extra:
        raise ValueError(
            "特征列与统一 14 维特征表不一致："
            f"缺失 {missing}；多余 {extra}"
        )
    return [c for c in FEATURE_NAMES if c in feat_cols]
