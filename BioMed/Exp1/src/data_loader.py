# -*- coding: utf-8 -*-
"""
实验一 · 数据加载与标签对齐

数据文件
--------
实验1-数据/PNAS_HCC.mat
    ma     : (1008, 122) float64  特征矩阵，**一行为一个特征，一列为一个样本**
    co     : (122, 1)    uint8    类别编码（见下方"标签陷阱"）
    sample : (122, 1)    object   样本ID

实验1-数据/PNAS_id_inforamtion.xlsx
    无表头三列：样本ID | 组名(healthy/HCC/Cirrhosis/HBV) | 编码(1/2/3/4)

标签陷阱（务必注意）
--------------------
.mat 中的 `co` **既不与 xlsx 的编码一致，也不与样本ID前缀一致**，实测交叉校验：

    co == 0  ->  HCC      (肝癌)      90 例   ← 注意是 0！
    co == 1  ->  healthy  (健康对照)   32 例

    xlsx 中却是  1 = healthy, 2 = HCC

即 `co` 与 xlsx 编码恰好相反。另外 `GM1100` 虽是 G 前缀，实为 HCC 样本，
因此**不能靠 ID 首字母判断类别**。

本项目统一约定：
    y = 1 - co  ->  1 = HCC(肝癌, 阳性) ,  0 = healthy(对照)
后续所有代码只用 y，不再直接使用 co。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

# 项目根目录（src/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "实验1-数据"

MAT_PATH = DATA_DIR / "PNAS_HCC.mat"
XLSX_PATH = DATA_DIR / "PNAS_id_inforamtion.xlsx"

# 类别名称：索引即 y 的取值
CLASS_NAMES = {0: "healthy", 1: "HCC"}
CLASS_NAMES_CN = {0: "健康对照", 1: "肝癌"}


@dataclass
class Exp1Data:
    """实验一数据集容器。

    Attributes
    ----------
    X_raw : ndarray, shape (n_samples, n_features)
        原始特征矩阵，已转置为"一行一个样本"。未做任何缩放。
    X : ndarray, shape (n_samples, n_features)
        按特征 z-score 标准化后的矩阵（mean=0, std=1）。
    y : ndarray, shape (n_samples,)
        类别标签，1 = HCC(肝癌)，0 = healthy(对照)。
    sample_ids : list[str]
        样本ID，顺序与 X / y 的行一一对应。
    feature_names : list[str]
        特征名。原数据无特征名，按 0..1007 编号。
    group_names : ndarray, shape (n_samples,)
        xlsx 中的原始组名，仅用于校验与展示。
    """

    X_raw: np.ndarray
    X: np.ndarray
    y: np.ndarray
    sample_ids: list
    feature_names: list
    group_names: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        n1 = int((self.y == 1).sum())
        n0 = int((self.y == 0).sum())
        return (
            f"样本数 {self.n_samples}（HCC {n1} / healthy {n0}），"
            f"特征数 {self.n_features}"
        )


def _load_label_table() -> pd.DataFrame:
    """读取 xlsx 标签表并补上列名（原文件无表头）。"""
    df = pd.read_excel(XLSX_PATH, header=None, names=["sample_id", "group", "code"])
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df


def load_exp1_data(standardize: bool = True, verbose: bool = True) -> Exp1Data:
    """加载实验一数据，并对齐、校验标签。

    Parameters
    ----------
    standardize : bool
        是否按特征做 z-score 标准化。PCA / t-SNE / 自编码器都对特征尺度敏感，
        而本数据各特征方差在 0.01~1.46 之间差异很大，故默认标准化。
        注意：实验内容3 的"方差法"必须用 X_raw（标准化后方差全为 1，无法排序）。
    verbose : bool
        是否打印校验信息。

    Returns
    -------
    Exp1Data

    Raises
    ------
    AssertionError
        当 .mat 与 .xlsx 的标签无法对齐或互相矛盾时抛出，避免静默错配。
    """
    mat = loadmat(MAT_PATH)

    # ---- 1. 特征矩阵：ma 是 (特征, 样本)，转置成 (样本, 特征) --------------
    ma = np.asarray(mat["ma"], dtype=np.float64)
    co = np.asarray(mat["co"]).ravel()
    sample_ids = [str(x[0]).strip() for x in mat["sample"].ravel()]

    assert ma.shape == (1008, 122), f"ma 形状异常: {ma.shape}"
    assert ma.shape[1] == len(co) == len(sample_ids), "ma 列数与标签数不一致"
    assert not np.isnan(ma).any(), "特征矩阵含 NaN"

    X_raw = ma.T.copy()  # (122, 1008)

    # ---- 2. 校验 co 的取值只能是 {0, 1} -----------------------------------
    assert set(np.unique(co).tolist()) <= {0, 1}, f"co 取值异常: {np.unique(co)}"

    # ---- 3. 与 xlsx 交叉校验 ---------------------------------------------
    df = _load_label_table()
    id2group = dict(zip(df["sample_id"], df["group"]))

    missing = [s for s in sample_ids if s not in id2group]
    assert not missing, f"以下样本在 xlsx 中缺失: {missing}"

    group_names = np.array([id2group[s] for s in sample_ids])

    # co == 0 应对应 HCC，co == 1 应对应 healthy。若此断言失败，
    # 说明数据版本变了，标签语义需要重新确认——不要放宽这个断言。
    expect = np.where(co == 0, "HCC", "healthy")
    mismatch = np.where(expect != group_names)[0]
    assert len(mismatch) == 0, (
        f"{len(mismatch)} 个样本的 co 与 xlsx 组名矛盾，例如 "
        f"{[(sample_ids[i], int(co[i]), group_names[i]) for i in mismatch[:5]]}"
    )

    # ---- 4. 统一标签：y = 1 - co，使 1 = HCC -------------------------------
    y = (1 - co).astype(np.int64)

    # ---- 5. 标准化 --------------------------------------------------------
    if standardize:
        X = StandardScaler().fit_transform(X_raw)
    else:
        X = X_raw.copy()

    feature_names = [f"F{i}" for i in range(X_raw.shape[1])]

    data = Exp1Data(
        X_raw=X_raw,
        X=X,
        y=y,
        sample_ids=sample_ids,
        feature_names=feature_names,
        group_names=group_names,
    )

    if verbose:
        print("=" * 68)
        print("数据加载完成")
        print(f"  文件        : {MAT_PATH.name} / {XLSX_PATH.name}")
        print(f"  特征矩阵 ma : {ma.shape} (特征 x 样本) -> X {X_raw.shape} (样本 x 特征)")
        print(f"  {data.summary()}")
        print(f"  标签对齐    : 122/122 样本与 xlsx 组名一致（co=0 -> HCC, co=1 -> healthy）")
        print(f"  标准化      : {'z-score (按特征)' if standardize else '无'}")
        print(f"  原始特征方差: {X_raw.var(axis=0).min():.4f} ~ {X_raw.var(axis=0).max():.4f}")
        print("=" * 68)

    return data


if __name__ == "__main__":
    load_exp1_data()
