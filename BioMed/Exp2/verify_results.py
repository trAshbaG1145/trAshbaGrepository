# -*- coding: utf-8 -*-
"""独立指标核对并重复运行；不是疾病分类性能的外部验证。"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    tables = sorted((OUT / "tables").glob("*.csv"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables}
    old = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    # 整次重跑包含所有 136 个配置；硬超时 180 秒。
    subprocess.run([sys.executable, str(ROOT / "exp2_main.py")],
                   cwd=ROOT, check=True, timeout=180)
    new = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables}
    assert before == after, "重跑 CSV 未能逐字节一致"
    old.pop("elapsed_seconds")
    new.pop("elapsed_seconds")
    assert old == new, "除运行耗时外的元数据不一致"
    metrics = pd.read_csv(OUT / "tables" / "all_metrics.csv")
    assignments = pd.read_csv(OUT / "tables" / "cluster_assignments.csv")
    quality = pd.read_csv(OUT / "tables" / "sample_quality.csv")
    arrays = np.load(OUT / "model_arrays.npz")
    y, z = arrays["truth"], arrays["PCA90"]
    d = pairwise_distances(z)
    np.fill_diagonal(d, 0)
    max_ri_error, max_sil_error = 0., 0.
    same_profiles = quality.profile_id.to_numpy()[:, None] == quality.profile_id.to_numpy()
    for row in metrics.itertuples():
        labels = assignments[row.config].to_numpy()
        assert len(np.unique(labels)) == row.k
        assert abs(adjusted_rand_score(y, labels) - row.ARI) < 1e-12
        # 独立计数 RI：同标签同簇或不同标签不同簇的无序样本对。
        same_class = y[:, None] == y[None, :]
        same_cluster = labels[:, None] == labels[None, :]
        pairs = np.triu_indices(len(y), 1)
        ri = np.mean((same_class == same_cluster)[pairs])
        max_ri_error = max(max_ri_error, abs(ri - row.RI))
        assert np.all(same_cluster[same_profiles]), "相同特征被拆入不同簇"
        # 独立 silhouette 公式；单点簇 silhouette 定义为 0。
        values = []
        for i in range(len(y)):
            own = labels == labels[i]
            if own.sum() == 1:
                values.append(0.)
                continue
            a = d[i, own].sum() / (own.sum() - 1)
            b = min(d[i, labels == g].mean() for g in np.unique(labels)
                    if g != labels[i])
            values.append((b - a) / max(a, b) if max(a, b) > 0 else 0.)
        max_sil_error = max(max_sil_error,
                            abs(np.mean(values) - row.silhouette_reference))
    assert max_ri_error < 1e-12 and max_sil_error < 1e-12
    for k in range(2, 9):
        c = arrays[f"consensus_k{k}"]
        assert np.allclose(c, c.T) and np.allclose(np.diag(c), 1)
        assert c.min() >= 0 and c.max() <= 1
        assert np.allclose(c[same_profiles], 1)
    assert len(metrics) == 42 + 70 + 3 + 21
    result = dict(status="VERIFIED", n_configs=len(metrics),
                  repeated_CSV_files=len(tables), CSV_byte_identical=True,
                  metadata_identical_except_elapsed=True,
                  max_RI_formula_error=max_ri_error,
                  max_silhouette_formula_error=max_sil_error,
                  identical_profiles_never_split=True,
                  consensus_symmetry_range_diagonal=True,
                  data_sha256=new["data_sha256"],
                  code_sha256=new["code_sha256"])
    (OUT / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
