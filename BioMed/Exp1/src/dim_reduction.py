# -*- coding: utf-8 -*-
"""
实验一 · 数据降维（实验内容2）

实现三种降维策略，统一接口：输入 (n_samples, n_features)，输出 (n_samples, n_components)。

    1. PCA          - 线性、无监督、确定性，主成分按方差贡献率排序
    2. t-SNE        - 非线性、无监督、随机（固定种子），擅长保留局部邻域结构
    3. AutoEncoder  - 非线性、无监督、随机（固定种子），以重构误差为目标的神经网络

三者都建立在"每个特征已 z-score 标准化"的矩阵上。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 全局随机种子，保证 t-SNE 与自编码器结果可复现
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. PCA
# ----------------------------------------------------------------------------
@dataclass
class PCAResult:
    embedding: np.ndarray          # (n_samples, n_components) 主成分得分
    explained_variance_ratio: np.ndarray
    n_components: int

    @property
    def cum_explained(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)


def pca_reduce(X: np.ndarray, n_components: int = 2) -> PCAResult:
    """PCA 降维。

    Parameters
    ----------
    X : (n_samples, n_features) 已标准化
    n_components : 保留的主成分个数。

    Returns
    -------
    PCAResult。`embedding` 的前 2 列即为可直接画散点图的前两个主成分。
    """
    n_components = int(min(n_components, min(X.shape)))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    embedding = pca.fit_transform(X)
    return PCAResult(
        embedding=embedding,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=n_components,
    )


def pca_full_spectrum(X: np.ndarray) -> np.ndarray:
    """返回全部主成分的方差贡献率（用于画碎石图 / 累计贡献率曲线）。"""
    n = int(min(X.shape))
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    pca.fit(X)
    return pca.explained_variance_ratio_


# ----------------------------------------------------------------------------
# 2. t-SNE
# ----------------------------------------------------------------------------
def tsne_reduce(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """t-SNE 降维。

    perplexity 可理解为"每个点考虑的有效近邻数"，是本方法最关键的参数，
    经验取值范围约 5~50，且必须 < n_samples。本实验会扫描 5/15/30 作对比。
    """
    perplexity = float(min(perplexity, max(2.0, X.shape[0] - 1) / 3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=n_iter,
            init="pca",          # pca 初始化比随机初始化更稳定
            learning_rate="auto",
            random_state=random_state,
        )
        return tsne.fit_transform(X)


# ----------------------------------------------------------------------------
# 3. 自编码器
# ----------------------------------------------------------------------------
class _AutoEncoderNet(nn.Module):
    """对称的欠完备自编码器：1008 -> 256 -> 64 -> latent -> 64 -> 256 -> 1008。

    隐层用 GELU；输出层不加激活（特征已标准化，取值可正可负）。
    """

    def __init__(self, n_features: int, latent_dim: int, hidden=(256, 64)):
        super().__init__()
        dims = [n_features, *hidden]
        enc, prev = [], dims[0]
        for h in dims[1:]:
            enc += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], latent_dim
        for h in reversed(dims[1:]):
            dec += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        dec.append(nn.Linear(prev, n_features))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass
class AEResult:
    embedding: np.ndarray                 # (n_samples, latent_dim)
    losses: list = field(default_factory=list)   # 每个 epoch 的训练 MSE
    final_recon_mse: float = float("nan")
    latent_dim: int = 2
    n_epochs: int = 0


def fit_autoencoder(
    X: np.ndarray,
    latent_dim: int = 2,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,
    hidden=(256, 64),
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
):
    """训练自编码器，返回 (model, AEResult)。

    单独抽出来是为了支持**留出集**评估：在训练集上拟合、在测试集上
    计算重构误差。仅看训练集重构误差会严重高估自编码器的能力——
    本网络约 50 万参数，而样本只有 122 个，足以把训练样本"背下来"。
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n = Xt.shape[0]

    model = _AutoEncoderNet(Xt.shape[1], latent_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss, n_batch = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1
        losses.append(epoch_loss / max(n_batch, 1))
        if verbose and (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch + 1:4d}/{n_epochs}  MSE={losses[-1]:.5f}")

    # 用最终模型对全部样本编码
    model.eval()
    with torch.no_grad():
        recon, z = model(Xt)
        final_mse = loss_fn(recon, Xt).item()

    result = AEResult(
        embedding=z.numpy(),
        losses=losses,
        final_recon_mse=final_mse,
        latent_dim=latent_dim,
        n_epochs=n_epochs,
    )
    return model, result


def autoencoder_reduce(X: np.ndarray, **kwargs) -> AEResult:
    """用欠完备自编码器降维，取**编码器输出**作为低维表示。

    注意：训练结束后模型处于 eval 模式（关闭 dropout 等），且本网络不含
    BatchNorm，因此整批前向与逐样本前向结果一致。
    """
    _, result = fit_autoencoder(X, **kwargs)
    return result


def ae_reconstruct_error(model, X_test: np.ndarray) -> float:
    """用已训练的自编码器在给定数据上计算重构 MSE。"""
    Xt = torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        recon, _ = model(Xt)
        return nn.MSELoss()(recon, Xt).item()
