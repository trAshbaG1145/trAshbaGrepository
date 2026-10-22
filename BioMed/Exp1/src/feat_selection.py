# -*- coding: utf-8 -*-
"""
实验一 · 特征筛选

包含两类策略，统一输出"得分越高越重要"的排序（降序）：

无监督（实验内容3，不看标签）
    1. variance_selection   - 特征在所有样本中的方差，降序
    2. entropy_selection    - 特征取值的 Shannon 熵（等宽分箱），降序

监督（实验内容4，使用标签）
    3. ttest_selection      - 两独立样本 t 检验（Welch），按 |t| 降序
    4. mannwhitney_selection- Mann-Whitney U 检验，按 |AUC-0.5| 降序
    5. mutual_info_selection- 特征与标签的互信息（连续特征 kNN 估计），降序
    6. mrmr_selection       - mRMR（最大相关最小冗余），MID / MIQ 两种准则
    7. lasso_selection      - L1 正则逻辑回归，按 |系数| 降序

设计说明
--------
* 方差法必须作用在**未标准化**的 X_raw 上——z-score 之后所有特征方差恒为 1，排序无意义。
* 熵法用等宽分箱。等宽分箱对特征的仿射变换是等变的（z-score 只是线性缩放平移），
  因此熵的结果在 X_raw 与 X 上完全一致，不受标准化影响。
* mRMR 按指导书步骤：先按各特征中值二值离散化，再在离散空间算互信息。
* Lasso 的 L1 会把大量系数压成精确的 0，导致大批特征并列。为保证排序可复现，
  以 |系数| 为主排序，并以 t 检验的 |t| 作为并列时的次序（在文档中已注明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


@dataclass
class SelectionResult:
    """一次特征筛选的结果。

    Attributes
    ----------
    name : 方法名
    scores : (n_features,) 得分，越大越重要
    ranking : (n_features,) 特征下标，按得分降序
    meta : 附加信息（p 值、非零系数个数等），供报告使用
    """

    name: str
    scores: np.ndarray
    ranking: np.ndarray
    meta: dict = field(default_factory=dict)

    def top(self, n: int) -> np.ndarray:
        """返回排名前 n 的特征下标（已按重要性降序）。"""
        return self.ranking[:n]


# ============================================================================
# 无监督方法
# ============================================================================
def variance_selection(X_raw: np.ndarray) -> SelectionResult:
    """方差法：计算每个特征在所有样本中的方差，降序排列。

    对应指导书实验内容3-A/3-B（Matlab function: VAR）。
    """
    scores = X_raw.var(axis=0, ddof=1)
    ranking = np.argsort(-scores, kind="stable")
    return SelectionResult("Variance", scores, ranking)


def shannon_entropy_per_feature(X: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """逐特征计算 Shannon 熵（等宽分箱后的分布熵）。

    特征 j 的取值范围均分为 n_bins 个等宽区间，统计落入各区间的样本数
    得到概率分布 p，熵 H = -Σ p·log2(p)。熵越大表示该特征取值分布越均匀、
    信息量/不确定性越大。

    返回归一化到 [0,1] 的熵（除以 log2(n_bins)），便于跨特征比较。
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    out = np.zeros(n_features, dtype=np.float64)
    max_entropy = np.log2(n_bins)

    for j in range(n_features):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[j] = 0.0          # 常数特征，无信息
            continue
        counts, _ = np.histogram(col, bins=n_bins, range=(lo, hi))
        p = counts / counts.sum()
        p = p[p > 0]              # 0·log0 约定为 0
        out[j] = -np.sum(p * np.log2(p)) / max_entropy

    return out


def entropy_selection(X: np.ndarray, n_bins: int = 20) -> SelectionResult:
    """熵法：按特征 Shannon 熵降序排列（熵高者视为信息量大）。"""
    scores = shannon_entropy_per_feature(X, n_bins=n_bins)
    ranking = np.argsort(-scores, kind="stable")
    return SelectionResult("Entropy", scores, ranking, meta={"n_bins": n_bins})


# ============================================================================
# 监督方法
# ============================================================================
def ttest_selection(X: np.ndarray, y: np.ndarray) -> SelectionResult:
    """t 检验：逐特征做两独立样本 t 检验（Welch，不假设方差齐性），按 |t| 降序。

    返回的 meta 含原始 p 值，并额外给出 Benjamini-Hochberg FDR 校正后的
    显著特征数（1008 个特征同时检验，不校正会有大量假阳性）。
    """
    pos = X[y == 1]
    neg = X[y == 0]
    t_stat, p_val = stats.ttest_ind(pos, neg, axis=0, equal_var=False)
    t_stat = np.nan_to_num(t_stat, nan=0.0)
    p_val = np.nan_to_num(p_val, nan=1.0)

    ranking = np.argsort(-np.abs(t_stat), kind="stable")

    # BH-FDR 校正
    order = np.argsort(p_val)
    m = len(p_val)
    bh = p_val[order] * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    q_val = np.empty(m)
    q_val[order] = np.clip(bh, 0, 1)

    return SelectionResult(
        "t-test",
        np.abs(t_stat),
        ranking,
        meta={
            "t": t_stat,
            "p": p_val,
            "q_bh": q_val,
            "n_p<0.05": int((p_val < 0.05).sum()),
            "n_q<0.05": int((q_val < 0.05).sum()),
        },
    )


def mannwhitney_selection(X: np.ndarray, y: np.ndarray) -> SelectionResult:
    """Mann-Whitney U 检验：非参数方法，不假设正态分布。

    直接按 p 值排序会遇到大量并列（尤其 p 极小值时精度受限），
    因此改用以效应量 |AUC-0.5| 作为得分——AUC = U/(n1·n0)，
    它与 p 值单调等价，但取值连续、无并列，更适合排序。
    AUC = 0.5 表示两组分布无差异，偏离越远越有判别力。
    """
    pos = X[y == 1]
    neg = X[y == 0]
    u_stat, p_val = stats.mannwhitneyu(pos, neg, axis=0, alternative="two-sided")

    n1, n0 = pos.shape[0], neg.shape[0]
    auc = u_stat / (n1 * n0)
    scores = np.abs(auc - 0.5)
    ranking = np.argsort(-scores, kind="stable")

    return SelectionResult(
        "Mann-Whitney U",
        scores,
        ranking,
        meta={"auc": auc, "p": p_val, "n_p<0.05": int((p_val < 0.05).sum())},
    )


def mutual_info_selection(X: np.ndarray, y: np.ndarray) -> SelectionResult:
    """互信息法：用 kNN 估计器计算每个特征与标签的互信息，降序。

    与相关系数不同，互信息能捕捉非线性依赖关系。
    """
    scores = mutual_info_classif(X, y, random_state=RANDOM_STATE, n_neighbors=5)
    ranking = np.argsort(-scores, kind="stable")
    return SelectionResult("Mutual Information", scores, ranking)


# ---------------------------------------------------------------------------
# mRMR
# ---------------------------------------------------------------------------
def _mi_from_binary_counts(n11, n10, n01, n00, n):
    """由 2x2 列联表的四个计数向量化计算互信息 I(X;Y)。

    对每个位置：I = Σ p(a,b)·ln(p(a,b) / (p(a)·p(b)))，约定 0·ln0 = 0。
    """
    n = float(n)
    p11, p10, p01, p00 = n11 / n, n10 / n, n01 / n, n00 / n
    p1_, p0_ = p11 + p10, p01 + p00          # X 的边缘分布
    p_1, p_0 = p11 + p01, p10 + p00          # Y 的边缘分布

    def term(pab, pa, pb):
        num = pab
        den = pa * pb
        # 只在 pab>0 且 den>0 时贡献，否则该格贡献 0
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where((num > 0) & (den > 0), num * np.log(num / den), 0.0)
        return val

    return (term(p11, p1_, p_1) + term(p10, p1_, p_0)
            + term(p01, p0_, p_1) + term(p00, p0_, p_0))


def _binarize_by_median(X: np.ndarray) -> np.ndarray:
    """按各特征在所有样本中的中值二值化（指导书实验内容4-A）。

    返回 int8 矩阵：值 > 中值记为 1，否则记为 0。
    """
    med = np.median(X, axis=0, keepdims=True)
    return (X > med).astype(np.int8)


def mrmr_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 200,
    criterion: str = "MID",
) -> SelectionResult:
    """mRMR：最大相关最小冗余（Peng et al., 2005）。

    步骤（对应指导书实验内容4-A/4-B）：
      1. 按各特征中值二值离散化；
      2. 取前 `n_features` 个特征做贪心前向选择，每一步在剩余特征中选
         使准则最大的一个加入已选集合 S：

           MID (Mutual Information Difference):
               max_f  I(f;c) - (1/|S|) * sum_{s in S} I(f;s)
           MIQ (Mutual Information Quotient):
               max_f  I(f;c) / ((1/|S|) * sum_{s in S} I(f;s) + eps)

         其中 I(f;c) 是"相关项"，I(f;s) 是"冗余项"。

    返回的 `scores` 为 `n_features` 个特征**被选中的先后顺序**（先选得分高），
    其余特征得分置 -inf 排在末尾。
    """
    if criterion not in ("MID", "MIQ"):
        raise ValueError("criterion 只能是 'MID' 或 'MIQ'")

    Xb = _binarize_by_median(X)
    n_samples, n_feat = Xb.shape
    n_features = int(min(n_features, n_feat))

    # ---- 相关项 I(f;c)：每个二值特征与二值标签的互信息（向量化） ----------
    c = (y == 1).astype(np.int8)
    n1_ = Xb.sum(axis=0).astype(np.float64)          # 每个特征中 1 的个数
    c1 = float(c.sum())
    n11 = (Xb * c[:, None]).sum(axis=0).astype(np.float64)
    relevance = _mi_from_binary_counts(
        n11, n1_ - n11, c1 - n11, n_samples - n1_ - c1 + n11, n_samples
    )

    # ---- 冗余项 I(f;s)：特征两两互信息矩阵（向量化，1008x1008） -----------
    n11_mat = Xb.astype(np.float64).T @ Xb.astype(np.float64)
    n1_col = n1_[:, None]
    n1_row = n1_[None, :]
    redundancy = _mi_from_binary_counts(
        n11_mat,
        n1_col - n11_mat,
        n1_row - n11_mat,
        n_samples - n1_col - n1_row + n11_mat,
        n_samples,
    )
    np.fill_diagonal(redundancy, 0.0)

    # ---- 贪婪前向选择 ------------------------------------------------------
    selected: list[int] = []
    available = np.ones(n_feat, dtype=bool)
    eps = 1e-12
    # 被选中特征得分 = 其准则值；未选中为 -inf
    order_scores = np.full(n_feat, -np.inf, dtype=np.float64)

    for _ in range(n_features):
        if len(selected) == 0:
            # 第一步无冗余项，直接取相关项最大者
            crit = np.where(available, relevance, -np.inf)
        else:
            red = redundancy[:, selected].mean(axis=1)
            if criterion == "MID":
                crit = relevance - red
            else:  # MIQ
                crit = relevance / (red + eps)
            crit = np.where(available, crit, -np.inf)

        best = int(np.argmax(crit))
        order_scores[best] = crit[best]
        available[best] = False
        selected.append(best)

    # 排序：先按被选中顺序，未被选中的按相关项降序补在后面
    sel_arr = np.array(selected)
    rest = np.array([i for i in range(n_feat) if i not in set(selected)])
    rest = rest[np.argsort(-relevance[rest], kind="stable")]
    ranking = np.concatenate([sel_arr, rest])

    return SelectionResult(
        f"mRMR-{criterion}",
        order_scores,
        ranking,
        meta={
            "criterion": criterion,
            "n_selected": n_features,
            "relevance": relevance,
            "selected_order": sel_arr,
        },
    )


def _fit_l1_logistic(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """拟合一个 L1 正则逻辑回归，返回系数向量（长度 = 特征数）。"""
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=C, max_iter=5000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    return clf.coef_.ravel()


def lasso_path_nonzero_counts(
    X: np.ndarray, y: np.ndarray,
    Cs=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
) -> dict:
    """扫描不同的 C，返回每个 C 下非零系数的特征个数（用于报告说明稀疏性）。"""
    return {float(c): int((np.abs(_fit_l1_logistic(X, y, c)) > 0).sum()) for c in Cs}


def lasso_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_alphas: int = 60,
    strict_C: float = 0.1,
) -> SelectionResult:
    """Lasso（L1 正则逻辑回归）特征筛选 —— 基于**正则化路径**排序。

    为什么不用单一 C 下的 |系数| 排序
    ---------------------------------
    L1 惩罚会把绝大多数特征的系数压成**精确的 0**。在本数据上取 C=0.05 时，
    1008 个特征里只有 3 个非零——即 1005 个特征得分完全相同（都是 0），
    排序完全由并列打破规则决定，并不能反映 Lasso 本身的选择倾向。

    本函数改用正则化路径（stability selection 的常用做法）。

    注意 C 的方向：sklearn 里 C 是正则强度的**倒数**，
        C 越小 -> 惩罚越强 -> 非零系数越少
        C 越大 -> 惩罚越弱 -> 非零系数越多
    因此 C 从小到大地扫，特征会依次"进入"模型。定义

        score_j = 特征 j **首次变为非零**时的那个 C

    越重要的特征越能扛住强惩罚，即 entry_C 越小、得分越高。这样 1008 个特征
    都能得到有区分度的得分，且完全不需要引入外部信息打破并列。

    Parameters
    ----------
    n_alphas : 正则路径上的 C 取值个数（几何间隔）。
    strict_C : 报告中"严格入选"特征集所用的 C，仅写入 meta 供说明稀疏性。

    Returns
    -------
    SelectionResult。meta 中记录路径上每个 C 的非零特征数、严格入选集等。
    """
    n_features = X.shape[1]

    def n_nonzero(c: float) -> int:
        return int((np.abs(_fit_l1_logistic(X, y, c)) > 0).sum())

    # ---- 指数搜索，框出路径两端 -------------------------------------------
    # 下端：惩罚强到没有特征进入
    c_min = 1.0
    for _ in range(60):
        if n_nonzero(c_min) == 0:
            break
        c_min /= 2.0
    # 上端：惩罚弱到几乎所有特征都进入
    c_max = 1.0
    for _ in range(60):
        if n_nonzero(c_max) >= 0.98 * n_features:
            break
        c_max *= 2.0

    Cs = np.geomspace(c_min, c_max, n_alphas)      # 由小到大，惩罚由强到弱

    # ---- 沿路径记录每个特征的首次进入点 ------------------------------------
    entry_C = np.full(n_features, np.inf, dtype=np.float64)
    entered = np.zeros(n_features, dtype=bool)
    path_counts = []
    for c in Cs:
        coef = np.abs(_fit_l1_logistic(X, y, float(c)))
        active = coef > 0
        newly = active & ~entered
        entry_C[newly] = float(c)                  # 首次进入，记下该 C
        entered |= active
        path_counts.append((float(c), int(active.sum())))

    never = ~np.isfinite(entry_C)

    # entry_C 越小越重要 -> 取负号作得分；从未进入者得 -inf 排最后
    scores = np.where(never, -np.inf, -entry_C)

    # 同一 C 步进入的特征 entry_C 相同，用该点 |系数| 大小打破并列（确定且可复现）
    coef_final = np.abs(_fit_l1_logistic(X, y, float(c_max)))
    ranking = np.lexsort((-coef_final, np.where(never, np.inf, entry_C)))

    coef_at_strict = _fit_l1_logistic(X, y, strict_C)
    strict_sel = np.where(np.abs(coef_at_strict) > 0)[0]

    return SelectionResult(
        "Lasso (L1 路径)",
        scores,
        ranking,
        meta={
            "strict_C": strict_C,
            "n_strict_nonzero": int(len(strict_sel)),
            "strict_features": strict_sel,
            "path": path_counts,
            "c_range": [float(c_min), float(c_max)],
            "n_never_selected": int(never.sum()),
            "entry_C": entry_C,
        },
    )
