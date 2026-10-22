# -*- coding: utf-8 -*-
"""实验二：CNV 聚类。运行 python Exp2/exp2_main.py；标签只用于评价。"""
from pathlib import Path
import hashlib
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.io import loadmat
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, pairwise_distances,
                             rand_score, silhouette_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"
SEED = 42
GROUPS = ["healthy", "HBV", "Cirrhosis", "HCC"]
COLORS = ["#4477AA", "#228833", "#CCBB44", "#CC6677"]


def save_table(frame, name):
    frame.to_csv(TAB / name, index=False, encoding="utf-8-sig")


def save_figure(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def pam(distance, candidates, k=4, repeats=8):
    """PAM SWAP：枚举每个 medoid 与非 medoid 的交换，取总距离最小者。
    候选来自不同原始特征向量，避免把重复样本作为多个初始中心。
    """
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(repeats):
        medoids = rng.choice(candidates, k, replace=False)
        cost = distance[:, medoids].min(axis=1).sum()
        for iteration in range(100):
            new_cost, new_medoids = cost, medoids.copy()
            for slot in range(k):
                other = np.delete(medoids, slot)
                nearest = distance[:, other].min(axis=1)
                costs = np.minimum(nearest[:, None], distance[:, candidates])
                costs = costs.sum(axis=0)
                costs[np.isin(candidates, medoids)] = np.inf
                j = int(np.argmin(costs))
                if costs[j] < new_cost - 1e-10:
                    new_cost = float(costs[j])
                    new_medoids = medoids.copy()
                    new_medoids[slot] = candidates[j]
            if new_cost >= cost - 1e-10:
                break
            medoids, cost = new_medoids, new_cost
        else:
            raise RuntimeError("PAM 达到迭代上限，请检查收敛")
        labels = distance[:, medoids].argmin(axis=1)
        if best is None or cost < best[0]:
            best = (float(cost), labels, medoids)
    return best


def consensus_runs(z, k, checkpoints=(50, 100, 200)):
    """80% 无放回重采样；同簇次数/共同出现次数，不把未抽到视为异簇。"""
    n = len(z)
    rng = np.random.default_rng(SEED + k)
    seen = np.zeros((n, n), dtype=np.int32)
    together = np.zeros_like(seen)
    outputs = {}
    for b in range(1, max(checkpoints) + 1):
        ix = np.sort(rng.choice(n, int(np.ceil(.8 * n)), replace=False))
        labels = KMeans(k, n_init=10, random_state=SEED + b).fit_predict(z[ix])
        block = np.ix_(ix, ix)
        seen[block] += 1
        together[block] += labels[:, None] == labels[None, :]
        if b in checkpoints:
            assert seen.min() > 0, "存在从未共同采样的样本对"
            c = together / seen
            np.fill_diagonal(c, 1)
            d = np.clip(1 - c, 0, 1)
            labels = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            ).fit_predict(d)
            upper = c[np.triu_indices(n, 1)]
            pac = float(np.mean((upper > .1) & (upper < .9)))
            outputs[b] = (labels, c.copy(), pac, int(seen.min()))
    return outputs


def main():
    start = time.perf_counter()
    threadpool_limits(1)  # 小样本单线程，避免过量线程开销并提高可重复性。
    for path in [FIG, TAB]:
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    data_path = ROOT / "实验2-数据" / "HCC_CNV_data.mat"
    m = loadmat(data_path)
    x = np.asarray(m["ma"], dtype=float).T
    sample = m["sample"]
    ids = [str(v[0]).strip() for v in sample[:, 0]]
    groups = np.array([str(v[0]).strip() for v in sample[:, 1]])
    assert x.shape == (225, 2475) and sample.shape == (225, 2)
    assert np.isfinite(x).all() and len(set(ids)) == len(ids)
    assert set(groups) == set(GROUPS)
    y = np.array([GROUPS.index(g) for g in groups])
    unique, candidates, inverse, counts = np.unique(
        x, axis=0, return_index=True, return_inverse=True, return_counts=True)
    all_two = np.all(x == 2, axis=1)
    qc = pd.DataFrame({"sample_id": ids, "group": groups,
                       "profile_id": inverse, "profile_size": counts[inverse],
                       "all_CNV_2": all_two, "altered_features": (x != 2).sum(1)})
    save_table(qc, "sample_quality.csv")
    group_qc = qc.groupby("group", sort=False).agg(
        n=("group", "size"), all_CNV_2=("all_CNV_2", "sum"),
        unique_profiles=("profile_id", "nunique"))
    save_table(group_qc.reset_index(), "group_quality.csv")
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(svd_solver="full").fit(scaled)
    scores = pca.transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dims = {str(t): int(np.searchsorted(cumulative, t) + 1)
            for t in [.8, .9, .95]}
    z = scores[:, :dims["0.9"]]
    reps = {"zscore_PCA80": scores[:, :dims["0.8"]],
            "zscore_PCA90": z, "zscore_PCA95": scores[:, :dims["0.95"]],
            "zscore_all": scaled,
            "raw_PCA90": PCA(.9, svd_solver="full").fit_transform(x)}
    top = np.argsort(-x.var(axis=0), kind="stable")[:500]
    reps["variance500_PCA90"] = PCA(.9, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(x[:, top]))
    save_table(pd.DataFrame({"PC": np.arange(1, len(cumulative) + 1),
                            "variance_ratio": pca.explained_variance_ratio_,
                            "cumulative": cumulative}), "pca_variance.csv")
    rows, partitions = [], {}
    reference_distance = pairwise_distances(z)

    def record(key, method, representation, k, labels, distmat, **extra):
        # 所有方案都在同一个 PCA90 欧氏空间复评，避免跨空间直接比较。
        assert len(labels) == len(x) and len(np.unique(labels)) == k
        partitions[key] = labels
        size = np.bincount(labels)
        row = dict(config=key, method=method, representation=representation,
                   k=k, silhouette_fit=silhouette_score(
                       distmat, labels, metric="precomputed"),
                   silhouette_reference=silhouette_score(
                       reference_distance, labels, metric="precomputed"),
                   DB_reference=davies_bouldin_score(z, labels),
                   RI=rand_score(y, labels), ARI=adjusted_rand_score(y, labels),
                   NMI=normalized_mutual_info_score(y, labels),
                   min_cluster=int(size.min()), max_cluster=int(size.max()),
                   cluster_sizes="/".join(map(str, sorted(size, reverse=True))))
        row.update(extra)
        rows.append(row)

    print("QC:", dict(group_qc["n"]), "unique profiles:", len(unique), flush=True)
    print("PCA dimensions:", dims, flush=True)
    for name, features in reps.items():
        distance = pairwise_distances(features)
        for k in range(2, 9):
            model = KMeans(k, n_init=50, random_state=SEED).fit(features)
            record(f"KM_{name}_k{k}", "KMeans", name, k,
                   model.labels_, distance, inertia=float(model.inertia_))
    print("KMeans sweep complete", flush=True)
    for metric in ["euclidean", "manhattan", "cosine"]:
        distance = pairwise_distances(z, metric=metric)
        linkages = ["single", "complete", "average"]
        if metric == "euclidean":
            linkages.append("ward")  # Ward 仅适用于欧氏距离。
        for mode in linkages:
            for k in range(2, 9):
                labels = AgglomerativeClustering(
                    n_clusters=k, metric=metric, linkage=mode).fit_predict(z)
                record(f"HC_{metric}_{mode}_k{k}", "Hierarchical",
                       "zscore_PCA90", k, labels, distance,
                       distance=metric, linkage=mode)
        objective, labels, medoids = pam(distance, candidates)
        record(f"PAM_{metric}_k4", "KMedoids", "zscore_PCA90", 4,
               labels, distance, distance=metric, objective=objective,
               medoid_ids="/".join(ids[i] for i in medoids))
    print("Hierarchy and PAM complete", flush=True)
    stability = []
    for n_init in [1, 10, 50]:
        runs = [KMeans(4, n_init=n_init, random_state=s).fit(z)
                for s in range(20)]
        pair_ari = [adjusted_rand_score(a.labels_, b.labels_)
                    for i, a in enumerate(runs) for b in runs[i + 1:]]
        stability.append(dict(n_init=n_init, seeds=20,
                              pair_ARI_mean=float(np.mean(pair_ari)),
                              pair_ARI_min=float(np.min(pair_ari)),
                              inertia_mean=float(np.mean([r.inertia_ for r in runs])),
                              inertia_std=float(np.std([r.inertia_ for r in runs]))))
    save_table(pd.DataFrame(stability), "kmeans_stability.csv")
    consensus = {}
    for k in range(2, 9):
        for b, (labels, c, pac, min_seen) in consensus_runs(z, k).items():
            key = f"CC_B{b}_k{k}"
            record(key, "Consensus", "zscore_PCA90", k, labels,
                   reference_distance, B=b, PAC=pac, min_pair_seen=min_seen)
            if b == 200:
                consensus[k] = c
        print(f"Consensus k={k} complete", flush=True)
    table = pd.DataFrame(rows)
    save_table(table, "all_metrics.csv")
    chosen = table.loc[table.k == 4].sort_values(
        ["ARI", "silhouette_reference", "config"], ascending=[False, False, True])
    best4 = chosen.iloc[0]
    best_sil = table.sort_values(
        ["silhouette_reference", "config"], ascending=[False, True]).iloc[0]
    family = chosen.groupby("method", sort=False).head(1)
    save_table(family, "best_k4_by_ARI.csv")
    # 一次组装宽表，避免 pandas 碎片化；簇编号仅用于显示，没有疾病语义。
    labels_df = pd.concat([qc[["sample_id", "group"]],
                          pd.DataFrame({k: v + 1 for k, v in partitions.items()})],
                         axis=1)
    save_table(labels_df, "cluster_assignments.csv")
    np.savez_compressed(OUT / "model_arrays.npz", PCA90=z, PC2=scores[:, :2],
                        truth=y, **{f"consensus_k{k}": c
                                    for k, c in consensus.items()})

    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    ax.plot(np.arange(1, 41), cumulative[:40], color=COLORS[0], lw=2)
    for t, dim in dims.items():
        ax.scatter(dim, cumulative[dim - 1], s=25)
        ax.annotate(f"{float(t):.0%}: {dim} PCs", (dim, cumulative[dim - 1]),
                    xytext=(5, -20), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Number of principal components", ylabel="Cumulative variance",
           ylim=(0, 1.03), xlim=(1, 40))
    save_figure(fig, "fig01_pca")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    for label, keys in [
        ("KMeans", [f"KM_zscore_PCA90_k{k}" for k in range(2, 9)]),
        ("Ward", [f"HC_euclidean_ward_k{k}" for k in range(2, 9)]),
        ("Consensus B=200", [f"CC_B200_k{k}" for k in range(2, 9)])]:
        sub = table.set_index("config").loc[keys]
        for ax, metric in zip(axes, ["silhouette_reference", "ARI"]):
            ax.plot(sub.k, sub[metric], marker="o", ms=3, label=label)
    for ax, title in zip(axes, ["Silhouette (common PCA90 space)", "ARI vs diseases"]):
        ax.set(xlabel="Number of clusters k", ylabel=title, xticks=range(2, 9))
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "fig02_k_sweep")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True, sharey=True)
    for ax, labels, title in zip(
        axes, [y, partitions[best4.config], partitions["CC_B200_k4"]],
        ["Known disease groups", "Best ARI at k=4", "Consensus k=4, B=200"]):
        for g in np.unique(labels):
            ix = labels == g
            text = GROUPS[g] if ax is axes[0] else f"Cluster {g + 1}"
            ax.scatter(scores[ix, 0], scores[ix, 1], s=18, alpha=.65,
                       color=COLORS[g], label=text, edgecolors="none")
        ax.set(title=title, xlabel="PC1")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("PC2")
    fig.tight_layout()
    save_figure(fig, "fig03_scatter")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, title in zip(axes, [best4.config, "CC_B200_k4"],
                              ["Best ARI at k=4", "Consensus k=4, B=200"]):
        ct = pd.crosstab(pd.Categorical(groups, categories=GROUPS),
                         partitions[key] + 1, dropna=False)
        save_table(ct.rename_axis("group").reset_index(), f"contingency_{key}.csv")
        ax.imshow(ct.values, cmap="Blues", vmin=0, vmax=90, aspect="auto")
        for i in range(4):
            for j in range(4):
                value = int(ct.iloc[i, j])
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > 50 else "black")
        ax.set(xticks=range(4), xticklabels=ct.columns,
               yticks=range(4), yticklabels=GROUPS, xlabel="Cluster", title=title)
    fig.tight_layout()
    save_figure(fig, "fig04_contingency")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    tree = linkage(z, method="ward", metric="euclidean")
    dendrogram(tree, truncate_mode="lastp", p=20, ax=ax,
               show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0, above_threshold_color=COLORS[0])
    ax.set(xlabel="Terminal groups (sample counts in parentheses)",
           ylabel="Ward merge distance")
    save_figure(fig, "fig05_dendrogram")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    c = consensus[4]
    order = np.lexsort((inverse, y, partitions["CC_B200_k4"]))
    heat = ax.imshow(c[np.ix_(order, order)], vmin=0, vmax=1,
                     cmap="Blues", interpolation="nearest")
    ax.set(xlabel="Samples ordered by consensus cluster",
           ylabel="Samples ordered by consensus cluster")
    fig.colorbar(heat, ax=ax, label="Co-clustering probability", fraction=.046)
    fig.tight_layout()
    save_figure(fig, "fig06_consensus")
    metadata = dict(seed=SEED, samples=len(x), features=x.shape[1],
                    group_counts=dict(zip(*np.unique(groups, return_counts=True))),
                    unique_profiles=len(unique), all_CNV_2=int(all_two.sum()),
                    constant_features=int((x.var(0) == 0).sum()),
                    PCA_dimensions=dims, PC2_variance=float(cumulative[1]),
                    PCA90_variance=float(cumulative[dims["0.9"] - 1]),
                    representations={k: v.shape[1] for k, v in reps.items()},
                    n_configs=len(table), best_k4_ARI=best4.to_dict(),
                    best_reference_silhouette=best_sil.to_dict(),
                    software=dict(python=platform.python_version(),
                                  numpy=np.__version__, scipy=scipy.__version__,
                                  sklearn=sklearn.__version__, pandas=pd.__version__,
                                  matplotlib=matplotlib.__version__),
                    data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    elapsed_seconds=time.perf_counter() - start)
    # 无需把 numpy 标量留在 JSON；未使用的指标写为 null。
    clean = json.loads(json.dumps(metadata, default=lambda v: v.item()))
    def no_nan(v):
        if isinstance(v, dict):
            return {k: no_nan(x) for k, x in v.items()}
        if isinstance(v, list):
            return [no_nan(x) for x in v]
        return None if isinstance(v, float) and not np.isfinite(v) else v
    (OUT / "metrics.json").write_text(
        json.dumps(no_nan(clean), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completed:", len(table), "configs; best k4 ARI:",
          best4.config, best4.ARI, flush=True)


if __name__ == "__main__":
    main()
