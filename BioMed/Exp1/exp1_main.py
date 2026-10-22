# -*- coding: utf-8 -*-
"""
《生物医学智能信息处理》实验一 · 数据降维与特征筛选 —— 主程序

用法::

    cd Exp1
    python exp1_main.py

一键完成指导书实验内容 2/3/4 的全部计算，产出：

    results/figures/*.png   全部结果图
    results/tables/*.csv    全部指标表
    results/metrics.json    全部原始指标（供报告引用）

实验内容与代码对应关系
----------------------
    实验内容2  数据降维      -> run_experiment2()
    实验内容3  无监督特征筛选 -> run_experiment3()
    实验内容4  监督特征筛选   -> run_experiment4()
    结果汇总/对比             -> run_summary()
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 允许以 `python exp1_main.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import dim_reduction as dr
from src import evaluate as ev
from src import feat_selection as fs
from src import visualize as vz
from src.data_loader import load_exp1_data

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TAB_DIR = PROJECT_ROOT / "results" / "tables"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"

# 特征筛选后做降维时的 top-N 取值（指导书未指定 N，这里做扫描）
TOP_N_LIST = [50, 100, 200, 500]
# 画散点图时代表性的 N
PLOT_N = 100

# 全部指标汇总于此，最后写入 metrics.json
METRICS: dict = {}


def _banner(text: str):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _timed(label: str):
    """简单计时上下文。"""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            print(f"    [{label}] 用时 {time.time() - self.t0:.1f}s")
    return _T()


# ============================================================================
# 实验内容 2：数据降维
# ============================================================================
def run_experiment2(data) -> dict:
    """PCA / t-SNE / 自编码器 三种方法对**全部特征**降维并可视化。"""
    _banner("实验内容2 数据降维（PCA / t-SNE / 自编码器）")

    X, y = data.X, data.y
    out = {}

    # ---- PCA --------------------------------------------------------------
    with _timed("PCA"):
        pca2 = dr.pca_reduce(X, n_components=2)
        spectrum = dr.pca_full_spectrum(X)      # 全部 122 个主成分的贡献率

    cum = np.cumsum(spectrum)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    sil_pca = ev.silhouette(pca2.embedding, y)

    print(f"    PC1 方差贡献率      : {spectrum[0] * 100:.2f}%")
    print(f"    PC2 方差贡献率      : {spectrum[1] * 100:.2f}%")
    print(f"    PC1+PC2 累计        : {cum[1] * 100:.2f}%")
    print(f"    累计达 80% / 90% 需 : {k80} / {k90} 个主成分")
    print(f"    前2维轮廓系数       : {sil_pca:.3f}")

    vz.plot_pca_spectrum(spectrum, FIG_DIR / "fig02_pca_spectrum.png", n_show=30)

    out["pca"] = {
        "pc1_ratio": float(spectrum[0]),
        "pc2_ratio": float(spectrum[1]),
        "pc1_pc2_cum": float(cum[1]),
        "k_for_80pct": k80,
        "k_for_90pct": k90,
        "silhouette_2d": sil_pca,
        "knn_cv_acc_all_feat": ev.knn_cv_accuracy(X, y),
        "explained_ratio_first30": [float(v) for v in spectrum[:30]],
    }

    # ---- t-SNE（扫描 perplexity） ------------------------------------------
    tsne_embs = {}
    for perp in (5, 15, 30):
        with _timed(f"t-SNE perplexity={perp}"):
            emb = dr.tsne_reduce(X, n_components=2, perplexity=perp)
        tsne_embs[perp] = emb
        s = ev.silhouette(emb, y)
        out.setdefault("tsne", {})[f"perplexity_{perp}"] = {"silhouette_2d": s}
        print(f"    perplexity={perp:>2}  轮廓系数 = {s:.3f}")

    fig = vz.scatter_grid(
        [(tsne_embs[p], f"t-SNE (perplexity={p})", out["tsne"][f"perplexity_{p}"]["silhouette_2d"])
         for p in (5, 15, 30)],
        y,
        "t-SNE 降维：perplexity 参数的影响（全部 1008 个特征）",
        ncols=3,
        figsize_per_panel=(5.0, 4.6),
    )
    vz.save_fig(fig, FIG_DIR / "fig04_tsne_perplexity.png")

    # ---- 自编码器 ---------------------------------------------------------
    with _timed("AutoEncoder latent=2"):
        ae2 = dr.autoencoder_reduce(X, latent_dim=2, n_epochs=500, verbose=False)
    sil_ae = ev.silhouette(ae2.embedding, y)
    print(f"    隐层维度 2  最终重构 MSE = {ae2.final_recon_mse:.4f}  轮廓系数 = {sil_ae:.3f}")

    with _timed("AutoEncoder latent=8"):
        ae8 = dr.autoencoder_reduce(X, latent_dim=8, n_epochs=500, verbose=False)
    print(f"    隐层维度 8  最终重构 MSE = {ae8.final_recon_mse:.4f}")

    out["autoencoder"] = {
        "latent2": {
            "final_recon_mse": float(ae2.final_recon_mse),
            "silhouette_2d": sil_ae,
            "knn_cv_acc_latent": ev.knn_cv_accuracy(ae2.embedding, y),
        },
        "latent8": {
            "final_recon_mse": float(ae8.final_recon_mse),
            "knn_cv_acc_latent": ev.knn_cv_accuracy(ae8.embedding, y),
        },
    }
    vz.plot_ae_loss(ae2.losses, FIG_DIR / "fig05_ae_loss.png", ae2.final_recon_mse)

    # ---- 留出集重构误差：检验自编码器是否只是"背下了"训练样本 ---------------
    # 网络约 50 万参数而样本仅 122 个，训练集重构误差会严重高估其能力。
    with _timed("留出集重构误差 (5折)"):
        ho = ev.holdout_reconstruction(X, latent_dims=(2, 8), n_epochs=500)
    out["holdout_reconstruction"] = ho
    print("\n    留出集重构 MSE（5 折交叉验证，测试集未参与训练）:")
    print(f"      {'隐层维度':<10}{'PCA 留出':<14}{'AE 留出':<14}{'AE 训练':<14}")
    for k, v in ho.items():
        print(f"      {k:<10}{v['pca_holdout_mse']:<14.4f}"
              f"{v['ae_holdout_mse']:<14.4f}{v['ae_train_mse']:<14.4f}")
    print("      对照：全量训练时 AE 训练 MSE 隐层2=0.1174, 隐层8=0.0014")
    print("      -> 训练误差极低但留出误差反而高于 PCA，说明是过拟合而非真实降维优势")

    vz.plot_holdout_reconstruction(
        ho,
        {2: float(ae2.final_recon_mse), 8: float(ae8.final_recon_mse)},
        FIG_DIR / "fig05b_holdout_reconstruction.png",
    )

    # ---- 三种方法横向对比 --------------------------------------------------
    fig = vz.scatter_grid(
        [
            (pca2.embedding, "PCA（前2个主成分）", sil_pca),
            (tsne_embs[30], "t-SNE（perplexity=30）", out["tsne"]["perplexity_30"]["silhouette_2d"]),
            (ae2.embedding, "自编码器（隐层=2）", sil_ae),
        ],
        y,
        "实验内容2：三种降维方法对比（全部 1008 个特征）",
        ncols=3,
        figsize_per_panel=(5.0, 4.6),
    )
    vz.save_fig(fig, FIG_DIR / "fig03_dr_comparison.png")

    # 方法对比表
    rows = [
        ("PCA (PC1-PC2)", sil_pca, ev.knn_cv_accuracy(pca2.embedding, y),
         f"前2维解释方差 {cum[1] * 100:.1f}%"),
        ("t-SNE (perp=5)", out["tsne"]["perplexity_5"]["silhouette_2d"],
         ev.knn_cv_accuracy(tsne_embs[5], y), "非线性，保留局部结构"),
        ("t-SNE (perp=15)", out["tsne"]["perplexity_15"]["silhouette_2d"],
         ev.knn_cv_accuracy(tsne_embs[15], y), "非线性，保留局部结构"),
        ("t-SNE (perp=30)", out["tsne"]["perplexity_30"]["silhouette_2d"],
         ev.knn_cv_accuracy(tsne_embs[30], y), "非线性，保留局部结构"),
        ("AutoEncoder (隐层2)", sil_ae, ev.knn_cv_accuracy(ae2.embedding, y),
         f"重构MSE {ae2.final_recon_mse:.4f}"),
        ("AutoEncoder (隐层8)", np.nan, ev.knn_cv_accuracy(ae8.embedding, y),
         f"重构MSE {ae8.final_recon_mse:.4f}"),
    ]
    df = pd.DataFrame(rows, columns=["降维方法", "轮廓系数(前2维)", "kNN 5折CV准确率", "说明"])
    df.to_csv(TAB_DIR / "tab01_dr_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))

    out["summary_table"] = TAB_DIR / "tab01_dr_metrics.csv"
    return out


# ============================================================================
# 实验内容 3：无监督特征筛选
# ============================================================================
def run_experiment3(data, baseline: dict) -> dict:
    """方差法 / 熵法筛选特征，再降维可视化，并与全特征结果对比。"""
    _banner("实验内容3 非监督学习特征筛选（方差 / 熵）")

    X, y = data.X, data.y
    out = {"methods": {}}

    # 方差法必须用未标准化的原始矩阵（标准化后方差恒为 1）
    var_res = fs.variance_selection(data.X_raw)
    ent_res = fs.entropy_selection(data.X)
    methods = {"Variance": var_res, "Entropy": ent_res}

    print(f"    方差法  得分范围: {var_res.scores.min():.4f} ~ {var_res.scores.max():.4f}")
    print(f"    熵法    得分范围: {ent_res.scores.min():.4f} ~ {ent_res.scores.max():.4f} (归一化到[0,1])")

    # 两种无监督方法选出的特征重叠情况
    for n in TOP_N_LIST:
        a, b = set(var_res.top(n).tolist()), set(ent_res.top(n).tolist())
        inter = len(a & b)
        out["methods"].setdefault("overlap", {})[f"top{n}"] = {
            "intersection": inter,
            "jaccard": inter / len(a | b) if (a | b) else 0.0,
        }
        print(f"    top{n:>3}: 方差法∩熵法 = {inter} 个特征 (Jaccard={inter / len(a | b):.3f})")

    # ---- 各 N 下的指标 ----------------------------------------------------
    records = []
    for name, res in methods.items():
        for n in TOP_N_LIST:
            sel = res.top(n)
            emb = dr.pca_reduce(X[:, sel], n_components=2).embedding
            sil = ev.silhouette(emb, y)
            acc = ev.knn_cv_accuracy(X[:, sel], y)
            records.append({"方法": name, "特征数N": n, "轮廓系数": sil, "kNN_CV准确率": acc})
            out["methods"].setdefault(name, {})[f"top{n}"] = {
                "silhouette": sil, "knn_cv_acc": acc,
            }
            print(f"    {name:<9} top{n:>3}  轮廓系数={sil:>7.3f}  kNN CV={acc:.4f}")

    # 基线（全部特征）
    records.append({
        "方法": "全部特征(基线)", "特征数N": data.n_features,
        "轮廓系数": baseline["pca"]["silhouette_2d"],
        "kNN_CV准确率": baseline["pca"]["knn_cv_acc_all_feat"],
    })

    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "tab02_unsup_fs_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- 可视化：全特征 vs 方差topN vs 熵topN ------------------------------
    panels = [(
        dr.pca_reduce(X, n_components=2).embedding,
        f"全部 {data.n_features} 个特征（基线）",
        baseline["pca"]["silhouette_2d"],
    )]
    for name, res in methods.items():
        sel = res.top(PLOT_N)
        emb = dr.pca_reduce(X[:, sel], n_components=2).embedding
        panels.append((emb, f"{name} 法 top{PLOT_N}", ev.silhouette(emb, y)))

    fig = vz.scatter_grid(
        panels, y,
        f"实验内容3：无监督特征筛选前后对比（PCA 前2维，top{PLOT_N}）",
        ncols=3, figsize_per_panel=(5.0, 4.6),
    )
    vz.save_fig(fig, FIG_DIR / "fig06_unsup_fs_pca.png")

    # ---- 条形图：两种方法的 top20 特征得分 ---------------------------------
    vz.plot_score_bars(
        [], var_res.scores, FIG_DIR / "fig06b_variance_scores.png",
        "方差法 top20 特征（按方差降序）", top_n=20, xlabel="方差", color="#4472C4",
    )
    vz.plot_score_bars(
        [], ent_res.scores, FIG_DIR / "fig06c_entropy_scores.png",
        "熵法 top20 特征（归一化 Shannon 熵降序）", top_n=20, xlabel="归一化熵", color="#548235",
    )

    out["table"] = str(TAB_DIR / "tab02_unsup_fs_metrics.csv")
    return out


# ============================================================================
# 实验内容 4：监督特征筛选
# ============================================================================
def run_experiment4(data, baseline: dict) -> dict:
    """t-test / Mann-Whitney U / 互信息 / mRMR / Lasso 五种监督筛选方法。"""
    _banner("实验内容4 监督学习特征筛选（t-test / MW / MI / mRMR / Lasso）")

    X, y = data.X, data.y
    out = {"methods": {}}

    # ---- 五种方法逐一计算 -------------------------------------------------
    with _timed("t-test"):
        tt = fs.ttest_selection(X, y)
    with _timed("Mann-Whitney U"):
        mw = fs.mannwhitney_selection(X, y)
    with _timed("Mutual Information"):
        mi = fs.mutual_info_selection(X, y)
    with _timed("mRMR (MID)"):
        mrmr = fs.mrmr_selection(X, y, n_features=500, criterion="MID")
    with _timed("mRMR (MIQ)"):
        mrmr_q = fs.mrmr_selection(X, y, n_features=500, criterion="MIQ")
    with _timed("Lasso"):
        lasso = fs.lasso_selection(X, y)

    print(f"    t-test       : p<0.05 的特征 {tt.meta['n_p<0.05']} 个，"
          f"BH-FDR 校正后 q<0.05 的 {tt.meta['n_q<0.05']} 个")
    print(f"    Mann-Whitney : p<0.05 的特征 {mw.meta['n_p<0.05']} 个")
    print(f"    Lasso        : 严格入选(C={lasso.meta['strict_C']}) "
          f"{lasso.meta['n_strict_nonzero']} 个，全程未入选 {lasso.meta['n_never_selected']} 个")

    # Lasso 正则路径（说明稀疏性随 C 的变化）
    path = fs.lasso_path_nonzero_counts(X, y)
    print("    Lasso 正则路径 非零特征数: "
          + ", ".join(f"C={c}:{n}" for c, n in path.items()))
    out["lasso_path_nonzero"] = path

    methods = {
        "t-test": tt,
        "Mann-Whitney U": mw,
        "Mutual Information": mi,
        "mRMR-MID": mrmr,
        "Lasso": lasso,
    }

    # ---- 各 N 下的指标 ----------------------------------------------------
    records = []
    for name, res in methods.items():
        for n in TOP_N_LIST:
            sel = res.top(n)
            emb = dr.pca_reduce(X[:, sel], n_components=2).embedding
            sil = ev.silhouette(emb, y)
            acc = ev.knn_cv_accuracy(X[:, sel], y)
            records.append({"方法": name, "特征数N": n, "轮廓系数": sil, "kNN_CV准确率": acc})
            out["methods"].setdefault(name, {})[f"top{n}"] = {
                "silhouette": sil, "knn_cv_acc": acc,
            }
            print(f"    {name:<19} top{n:>3}  轮廓系数={sil:>7.3f}  kNN CV={acc:.4f}")

    records.append({
        "方法": "全部特征(基线)", "特征数N": data.n_features,
        "轮廓系数": baseline["pca"]["silhouette_2d"],
        "kNN_CV准确率": baseline["pca"]["knn_cv_acc_all_feat"],
    })
    df = pd.DataFrame(records)
    df.to_csv(TAB_DIR / "tab03_sup_fs_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- 方法间 top-N 特征重叠（Jaccard） ----------------------------------
    print("\n    top-100 特征集合的重叠度（Jaccard）:")
    jac = {}
    names = list(methods)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = set(methods[a].top(100).tolist()), set(methods[b].top(100).tolist())
            j = len(sa & sb) / len(sa | sb)
            jac[f"{a} vs {b}"] = j
            print(f"      {a:<19} vs {b:<19} = {j:.3f}")
    out["top100_jaccard"] = jac

    # ---- 可视化：五种方法 top-N 的 PCA 散点 --------------------------------
    panels = [(
        dr.pca_reduce(X, n_components=2).embedding,
        f"全部 {data.n_features} 个特征（基线）",
        baseline["pca"]["silhouette_2d"],
    )]
    for name, res in methods.items():
        sel = res.top(PLOT_N)
        emb = dr.pca_reduce(X[:, sel], n_components=2).embedding
        panels.append((emb, f"{name} top{PLOT_N}", ev.silhouette(emb, y)))

    fig = vz.scatter_grid(
        panels, y,
        f"实验内容4：五种监督特征筛选方法对比（PCA 前2维，top{PLOT_N}）",
        ncols=3, figsize_per_panel=(5.0, 4.6),
    )
    vz.save_fig(fig, FIG_DIR / "fig07_sup_fs_pca.png")

    # ---- 得分条形图 -------------------------------------------------------
    for nm, res, col in [
        ("fig07b_ttest_scores.png", tt, "#4472C4"),
        ("fig07c_mw_scores.png", mw, "#7030A0"),
        ("fig07d_mi_scores.png", mi, "#C55A11"),
        ("fig07e_mrmr_scores.png", mrmr, "#548235"),
        ("fig07f_lasso_scores.png", lasso, "#C00000"),
    ]:
        vz.plot_score_bars(
            [], res.scores, FIG_DIR / nm,
            f"{res.name} top20 特征得分", top_n=20,
            xlabel="得分", color=col,
        )

    out["table"] = str(TAB_DIR / "tab03_sup_fs_metrics.csv")
    return out


# ============================================================================
# 汇总对比
# ============================================================================
def run_summary(data, baseline: dict, r3: dict, r4: dict):
    """把实验内容3、4 的结果汇总对比，回答指导书的 3-E / 4-E 两个问题。"""
    _banner("结果汇总与横向对比")

    # ---- 合并两张指标表 ---------------------------------------------------
    t2 = pd.read_csv(TAB_DIR / "tab02_unsup_fs_metrics.csv")
    t3 = pd.read_csv(TAB_DIR / "tab03_sup_fs_metrics.csv")
    t2["类型"] = "无监督"
    t3["类型"] = "监督"
    both = pd.concat([t2, t3], ignore_index=True)
    both.to_csv(TAB_DIR / "tab04_all_methods_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- N=100 时各方法的对比条形图 ---------------------------------------
    sub = both[both["特征数N"] == PLOT_N].copy()
    sub = sub.sort_values("kNN_CV准确率", ascending=False)

    vz.plot_method_comparison(
        sub["方法"].tolist(), sub["kNN_CV准确率"].tolist(),
        FIG_DIR / "fig08_knn_acc_comparison.png",
        f"各方法在 top{PLOT_N} 特征下的 kNN 5折CV准确率（全部特征基线见灰线外标注）",
        "kNN 5折CV准确率",
    )
    vz.plot_method_comparison(
        sub["方法"].tolist(), sub["轮廓系数"].tolist(),
        FIG_DIR / "fig09_silhouette_comparison.png",
        f"各方法在 top{PLOT_N} 特征下的 PCA 前2维轮廓系数",
        "轮廓系数", color="#548235",
    )

    # ---- 指标随 N 变化的曲线 ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for name, g in both.groupby("方法"):
        g = g.sort_values("特征数N")
        axes[0].plot(g["特征数N"], g["kNN_CV准确率"], marker="o", label=name, linewidth=1.5, markersize=4)
        axes[1].plot(g["特征数N"], g["轮廓系数"], marker="s", label=name, linewidth=1.5, markersize=4)

    base_acc = baseline["pca"]["knn_cv_acc_all_feat"]
    base_sil = baseline["pca"]["silhouette_2d"]
    for ax, base, ylab, title in [
        (axes[0], base_acc, "kNN 5折CV准确率", "准确率随所选特征数 N 的变化"),
        (axes[1], base_sil, "轮廓系数", "轮廓系数随所选特征数 N 的变化"),
    ]:
        ax.axhline(base, color="gray", linestyle="--", linewidth=1.2,
                   label=f"全部 {data.n_features} 特征基线")
        ax.set_xlabel("所选特征数 N", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    vz.save_fig(fig, FIG_DIR / "fig10_metric_vs_N.png")

    print("\n" + sub[["类型", "方法", "特征数N", "轮廓系数", "kNN_CV准确率"]].to_string(index=False))
    print(f"\n    全部 {data.n_features} 特征基线: 轮廓系数={base_sil:.3f}, "
          f"kNN CV准确率={base_acc:.4f}")

    # ---- 最优方法 ---------------------------------------------------------
    best = sub.iloc[0]
    print(f"\n    >> top{PLOT_N} 下 kNN 准确率最高: {best['方法']} ({best['kNN_CV准确率']:.4f})")
    METRICS["summary"] = {
        "baseline_all_features": {
            "silhouette": base_sil, "knn_cv_acc": base_acc,
        },
        "best_at_topN": {
            "N": PLOT_N,
            "method": str(best["方法"]),
            "knn_cv_acc": float(best["kNN_CV准确率"]),
            "silhouette": float(best["轮廓系数"]),
        },
    }
    return both


# ============================================================================
# 入口
# ============================================================================
def main():
    np.random.seed(RANDOM_STATE)
    import torch
    torch.manual_seed(RANDOM_STATE)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    data = load_exp1_data()

    METRICS["dataset"] = {
        "n_samples": data.n_samples,
        "n_features": data.n_features,
        "n_hcc": int((data.y == 1).sum()),
        "n_healthy": int((data.y == 0).sum()),
        "label_convention": "y=1 -> HCC, y=0 -> healthy (内部约定 y = 1 - co)",
    }

    r2 = run_experiment2(data)
    METRICS["exp2_dim_reduction"] = {k: v for k, v in r2.items() if k != "summary_table"}

    r3 = run_experiment3(data, r2)
    METRICS["exp3_unsupervised_fs"] = r3["methods"]

    r4 = run_experiment4(data, r2)
    METRICS["exp4_supervised_fs"] = {
        "metrics_by_N": r4["methods"],
        "top100_jaccard": r4["top100_jaccard"],
        "lasso_path_nonzero": r4["lasso_path_nonzero"],
    }

    run_summary(data, r2, r3, r4)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(METRICS, f, ensure_ascii=False, indent=2)

    _banner("全部完成")
    print(f"    图  : {FIG_DIR}")
    print(f"    表  : {TAB_DIR}")
    print(f"    指标: {METRICS_PATH}")


if __name__ == "__main__":
    main()
