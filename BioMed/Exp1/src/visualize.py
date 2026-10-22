# -*- coding: utf-8 -*-
"""
实验一 · 可视化

统一散点图风格：健康对照(healthy) 用蓝色圆点，肝癌(HCC) 用红色三角，
图题标注轮廓系数，便于横向比较不同降维/筛选方法的类别分离效果。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无界面后端，服务器/脚本环境下也能出图
import matplotlib.pyplot as plt
import numpy as np

# ---- 中文字体配置（Windows 自带字体） --------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

# 类别样式：索引为 y 的取值
STYLE = {
    0: dict(label="健康对照 (healthy)", color="#2E75B6", marker="o", alpha=0.75),
    1: dict(label="肝癌 (HCC)",        color="#C00000", marker="^", alpha=0.75),
}


def scatter_2d(
    embedding: np.ndarray,
    y: np.ndarray,
    title: str,
    ax: plt.Axes | None = None,
    silhouette_value: float | None = None,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    annotate: bool = True,
):
    """在二维嵌入上按类别画散点图。

    Parameters
    ----------
    embedding : (n_samples, >=2) 低维表示，取前两列作图
    y : 类别标签，1 = HCC，0 = healthy
    title : 图题
    ax : 已有坐标轴；为 None 时新建
    silhouette_value : 若给出，附加到图题上
    annotate : 是否在图题后附加轮廓系数
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.2))

    embedding = np.asarray(embedding)
    y = np.asarray(y)

    for cls in (0, 1):
        m = y == cls
        if not m.any():
            continue
        st = STYLE[cls]
        ax.scatter(
            embedding[m, 0], embedding[m, 1],
            label=f"{st['label']} (n={int(m.sum())})",
            c=st["color"], marker=st["marker"], alpha=st["alpha"],
            s=44, edgecolors="white", linewidths=0.6,
        )

    if annotate and silhouette_value is not None and np.isfinite(silhouette_value):
        title = f"{title}\n轮廓系数 silhouette = {silhouette_value:.3f}"

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=8)
    return ax


def save_fig(fig, path: str | Path, close: bool = True):
    """保存图片并确保目录存在。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


def scatter_grid(
    panels: list[tuple[np.ndarray, str, float | None]],
    y: np.ndarray,
    suptitle: str,
    ncols: int = 3,
    figsize_per_panel=(5.4, 4.8),
):
    """把多张散点图排成网格，用于方法间的横向对比。

    Parameters
    ----------
    panels : [(embedding, title, silhouette), ...]
    """
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    w, h = figsize_per_panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (emb, title, sil) in zip(axes, panels):
        scatter_2d(emb, y, title, ax=ax, silhouette_value=sil)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=13, y=1.002)
    fig.tight_layout()
    return fig


def plot_pca_spectrum(
    explained_ratio: np.ndarray,
    path: str | Path,
    n_show: int = 30,
):
    """画 PCA 碎石图 + 累计方差贡献率曲线（对应指导书实验内容2-B）。"""
    n_show = int(min(n_show, len(explained_ratio)))
    cum = np.cumsum(explained_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].bar(np.arange(1, n_show + 1), explained_ratio[:n_show] * 100,
                color="#4472C4", alpha=0.85)
    axes[0].set_xlabel("主成分序号", fontsize=9)
    axes[0].set_ylabel("方差贡献率 (%)", fontsize=9)
    axes[0].set_title(f"前 {n_show} 个主成分的方差贡献率", fontsize=11)
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

    axes[1].plot(np.arange(1, len(cum) + 1), cum * 100, color="#C00000", linewidth=1.8)
    for thr in (80, 90):
        if cum[-1] * 100 >= thr:
            k = int(np.searchsorted(cum, thr / 100) + 1)
            axes[1].axhline(thr, color="gray", linestyle=":", linewidth=0.9)
            axes[1].annotate(f"{thr}% 需 {k} 个主成分", xy=(k, thr),
                             xytext=(k + 2, thr - 9), fontsize=8,
                             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    axes[1].set_xlabel("主成分个数", fontsize=9)
    axes[1].set_ylabel("累计方差贡献率 (%)", fontsize=9)
    axes[1].set_title("累计方差贡献率", fontsize=11)
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_ae_loss(losses: list, path: str | Path, final_mse: float):
    """画自编码器训练损失曲线。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#548235", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("重构 MSE (log)", fontsize=9)
    ax.set_title(f"自编码器训练损失曲线（最终 MSE = {final_mse:.4f}）", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_score_bars(
    names: list[str],
    scores: np.ndarray,
    path: str | Path,
    title: str,
    top_n: int = 20,
    xlabel: str = "得分",
    color: str = "#4472C4",
):
    """画某方法下 top-N 特征的得分条形图。"""
    scores = np.asarray(scores)
    order = np.argsort(-scores)[:top_n]
    labels = [f"F{i}" for i in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(np.arange(len(order))[::-1], scores[order], color=color, alpha=0.85)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="x")
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_holdout_reconstruction(
    holdout: dict,
    full_train_mse: dict,
    path: str | Path,
):
    """对比 PCA / 自编码器 的重构误差：训练集 vs 留出集。

    Parameters
    ----------
    holdout : {k: {"pca_holdout_mse", "ae_holdout_mse", "ae_train_mse"}}
    full_train_mse : {k: 在全量数据上训练时的 AE 训练 MSE}，用于凸显过拟合
    """
    ks = sorted(holdout.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(5.6 * len(ks), 4.4), squeeze=False)
    axes = axes.ravel()

    for ax, k in zip(axes, ks):
        v = holdout[k]
        labels = ["PCA\n(留出集)", "自编码器\n(留出集)", "自编码器\n(训练集)"]
        vals = [v["pca_holdout_mse"], v["ae_holdout_mse"], v["ae_train_mse"]]
        if k in full_train_mse:
            labels.append("自编码器\n(全量训练集)")
            vals.append(full_train_mse[k])

        colors = ["#4472C4", "#C00000", "#E8A33D", "#B0B0B0"][: len(vals)]
        bars = ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.9)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("重构 MSE（越小越好）", fontsize=9)
        ax.set_title(f"隐层/主成分维度 k = {k}", fontsize=11)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("重构能力对比：训练集误差低不代表泛化好（自编码器过拟合）", fontsize=12.5)
    fig.tight_layout()
    save_fig(fig, path)
    return path


def plot_method_comparison(
    method_names: list[str],
    metric_values: list[float],
    path: str | Path,
    title: str,
    ylabel: str,
    color: str = "#4472C4",
    highlight_best: bool = True,
):
    """画不同方法在某个指标上的对比条形图。"""
    vals = np.asarray(metric_values, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(method_names)), 4.4))

    colors = [color] * len(vals)
    if highlight_best and np.isfinite(vals).any():
        colors[int(np.nanargmax(vals))] = "#C00000"

    bars = ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.88)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(method_names, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    save_fig(fig, path)
    return path
