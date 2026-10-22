# -*- coding: utf-8 -*-
"""
实验一 · 降维/筛选结果的量化评估

指导书只要求"用散点图看样本是否有分布模式"，但肉眼比较不同方法的图
缺乏说服力。这里补充两个客观指标，让"特征筛选是否有利于区分两类"
这个结论有数字支撑：

1. silhouette_score —— 轮廓系数，衡量降维后二维空间中两类的分离程度。
   取值 [-1, 1]，越接近 1 表示类内越紧凑、类间越分散，-1 表示严重重叠。

2. knn_cv_accuracy —— 在**所选特征子集**上做 kNN 分层交叉验证的准确率，
   作为该特征集"判别能力"的代理指标。降维方法本身不产生分类器，
   该指标衡量的是特征携带的类别信息量。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def silhouette(embedding: np.ndarray, y: np.ndarray) -> float:
    """二维嵌入的轮廓系数。样本数或类别数不足时返回 nan。"""
    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        return float("nan")
    if len(np.unique(y)) < 2 or embedding.shape[0] <= len(np.unique(y)):
        return float("nan")
    # 只看前两维（画散点图用的就是这两维）
    return float(silhouette_score(embedding[:, :2], y))


def knn_cv_accuracy(
    X_sub: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    k: int = 5,
    n_repeats: int = 5,
) -> float:
    """在给定特征子集上做分层 k 折交叉验证，返回平均准确率。

    使用 ``n_repeats`` 次不同随机划分求平均，降低单次划分带来的波动。
    管道内先标准化，避免每次划分的尺度泄漏。
    5 折下若某折恰好取到很少的 HCC 样本，准确率方差会较大，故重复取平均。
    """
    X_sub = np.asarray(X_sub)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))

    scores = []
    for r in range(n_repeats):
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + r
        )
        scores.append(cross_val_score(pipe, X_sub, y, cv=cv, scoring="accuracy").mean())
    return float(np.mean(scores))


def evaluate_subset(
    X_sub: np.ndarray,
    y: np.ndarray,
    reducer=None,
    reducer_kwargs: dict | None = None,
) -> dict:
    """对一个特征子集同时给出轮廓系数与 kNN 交叉验证准确率。

    Parameters
    ----------
    X_sub : (n_samples, n_selected) 已选特征
    y : 标签
    reducer : 可调用对象，签名 reducer(X, **kwargs) -> 二维嵌入。
        若为 None，则用 PCA 前两维。
    """
    if reducer is None:
        from .dim_reduction import pca_reduce
        reducer = lambda X, **kw: pca_reduce(X, n_components=2).embedding

    emb = reducer(X_sub, **(reducer_kwargs or {}))
    return {
        "silhouette": silhouette(emb, y),
        "knn_cv_acc": knn_cv_accuracy(X_sub, y),
    }


def holdout_reconstruction(
    X: np.ndarray,
    latent_dims=(2, 8),
    n_splits: int = 5,
    n_epochs: int = 500,
    random_state: int = RANDOM_STATE,
) -> dict:
    """比较 PCA 与自编码器的**留出集**重构误差。

    为什么必须看留出集
    ------------------
    自编码器的参数量（本实验约 50 万）远超样本量（122），在训练集上可以把
    样本几乎完全"背下来"。实测在全部样本上训练时，隐层 8 维的训练重构 MSE
    低至 0.0014，看上去远优于线性 PCA——但这是过拟合的假象。

    本函数在 5 折交叉验证下，每折都用**训练集**拟合模型、在**未参与训练的
    测试集**上计算重构 MSE，得到的才是可比的泛化重构能力。

    返回
    ----
    {k: {"pca_holdout_mse":..., "ae_holdout_mse":..., "ae_train_mse":...}}
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold

    from .dim_reduction import fit_autoencoder, ae_reconstruct_error

    X = np.asarray(X, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out = {}

    for k in latent_dims:
        pca_ho, ae_ho, ae_tr = [], [], []
        for tr, te in kf.split(X):
            pca = PCA(n_components=k, random_state=random_state).fit(X[tr])
            pca_ho.append(float(((X[te] - pca.inverse_transform(pca.transform(X[te]))) ** 2).mean()))

            model, _ = fit_autoencoder(
                X[tr], latent_dim=k, n_epochs=n_epochs, random_state=0
            )
            ae_ho.append(ae_reconstruct_error(model, X[te]))
            ae_tr.append(ae_reconstruct_error(model, X[tr]))

        out[int(k)] = {
            "pca_holdout_mse": float(np.mean(pca_ho)),
            "ae_holdout_mse": float(np.mean(ae_ho)),
            "ae_train_mse": float(np.mean(ae_tr)),
        }

    return out
