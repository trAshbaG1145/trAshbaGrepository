# -*- coding: utf-8 -*-
"""
生成《实验一 简要版报告》的代码附录。

代码直接从 src/ 与 exp1_main.py 的真实源文件中**按名字抽取**，
不手工誊抄，保证附录里的代码与实际跑出结果的代码一致。

附录被组织成一个**可独立运行的完整脚本**：导入 → 常量 → 数据结构 →
辅助函数 → 各模块核心函数 → 主程序。把生成结果另存为
``Exp1/实验一_附录代码.py`` 并执行，即可复现报告中的全部结果。

用法::

    cd Exp1/report
    python gen_brief.py            # 生成报告
    python gen_brief.py --check    # 生成后追加可复现性自检
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

BODY = HERE / "实验一_报告_简要版_正文.md"
OUT = HERE / "实验一_报告_简要版.md"
MARKER = "<!-- CODE_APPENDIX -->"

# 附录组装完成后落在 Exp1/ 下，用于可复现性自检
STANDALONE_NAME = "实验一_附录代码.py"


# ---------------------------------------------------------------------------
# 从源码抽取
# ---------------------------------------------------------------------------
def _seg(lines: list[str], node) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(lines[start - 1:node.end_lineno]).rstrip("\n")


def extract_defs(path: pathlib.Path, names: list[str]) -> list[tuple[str, str]]:
    """抽取顶层函数/类，返回 [(名字, 源码)]，保持文件内原始顺序。"""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    wanted = set(names)
    found: dict[str, tuple[int, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in wanted:
                found[node.name] = (node.lineno, _seg(lines, node))
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"{path.name} 中找不到: {sorted(missing)}")
    return [(n, s) for n, (_, s) in sorted(found.items(), key=lambda kv: kv[1][0])]


def extract_assignments(path: pathlib.Path, names: list[str]) -> list[tuple[str, str]]:
    """抽取模块级简单赋值（常量），返回 [(名字, 源码)]。"""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out.append((t.id, _seg(lines, node)))
    return out


def extract_imports(paths: list[pathlib.Path]) -> list[str]:
    """收集模块级 import 语句，去重并排除相对导入与 src.* 自引用。"""
    seen: list[str] = []
    for path in paths:
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    # 跳过相对导入与包内自引用（拼成单文件后不存在这些模块）
                    if node.level > 0 or mod.startswith("src"):
                        continue
                stmt = _seg(lines, node)
                if stmt not in seen:
                    seen.append(stmt)
    return seen


def strip_relative_imports(seg: str) -> tuple[str, int]:
    """去掉函数体内的相对导入。

    拼成单文件后 ``from .dim_reduction import xxx`` 无法解析，但这些名字
    在同一文件顶层已经定义，直接删除该行即可。
    """
    out, n = [], 0
    for line in seg.splitlines():
        if re.match(r"\s*from\s+\.\w*\s+import\s", line) or re.match(r"\s*from\s+\.\w+\.\w+\s+import\s", line):
            n += 1
            continue
        out.append(line)
    return "\n".join(out), n


def code_block(seg: str) -> str:
    return "```python\n" + seg + "\n```\n"


# ---------------------------------------------------------------------------
# 附录各段
# ---------------------------------------------------------------------------
MODULE_SPEC: list[tuple[str, str, list[str]]] = [
    ("src/data_loader.py", "数据加载与标签对齐",
     ["load_exp1_data"]),
    ("src/dim_reduction.py", "降维：PCA / t-SNE / 自编码器",
     ["pca_reduce", "pca_full_spectrum", "tsne_reduce", "_AutoEncoderNet",
      "fit_autoencoder", "autoencoder_reduce"]),
    ("src/feat_selection.py", "特征筛选：无监督 + 监督",
     ["variance_selection", "shannon_entropy_per_feature", "entropy_selection",
      "ttest_selection", "mannwhitney_selection", "mutual_info_selection",
      "_mi_from_binary_counts", "_binarize_by_median", "mrmr_selection",
      "lasso_path_nonzero_counts", "lasso_selection"]),
    ("src/evaluate.py", "评估：轮廓系数 / kNN 交叉验证 / 留出集重构误差",
     ["silhouette", "knn_cv_accuracy", "holdout_reconstruction"]),
]

# 绘图模块不列入报告附录（非算法核心，约 250 行，为控制篇幅省略）。
# 但在自检时会被追加到可运行副本里，以验证附录代码能端到端复现结果。
VERIFY_ONLY_FILES = ["src/visualize.py"]

INTRO = {
    "src/data_loader.py": (
        "对应指导书：实验数据读取。**关键点是标签对齐**——`co` 与 xlsx 编码相反，\n"
        "代码用断言逐样本校验，不一致直接报错而非静默错配。\n"
    ),
    "src/dim_reduction.py": (
        "对应指导书：实验内容 2。三种降维方法统一接口。\n"
        "`fit_autoencoder` 单独抽出来是为了支持留出集评估（只返回 `AEResult` 的\n"
        "`autoencoder_reduce` 无法在测试集上算重构误差）。\n"
    ),
    "src/feat_selection.py": (
        "对应指导书：实验内容 3（方差、熵）与实验内容 4（t 检验、MW、互信息、mRMR、Lasso）。\n"
        "所有方法统一返回 `SelectionResult`（得分 + 排序）。\n"
        "`_mi_from_binary_counts` 是 mRMR 的核心：把特征两两互信息从约 50 万次\n"
        "单独调用变成一次 `2x2` 列联表的向量化计算。\n"
    ),
    "src/evaluate.py": (
        "指导书只要求看散点图，这里补充量化指标。\n"
        "`holdout_reconstruction` 是发现自编码器过拟合的关键——只在训练集上算\n"
        "重构误差会得出完全相反的结论。\n"
    ),
    "src/visualize.py": (
        "绘图模块（未列入附录，见 `src/visualize.py`）。\n"
    ),
}


def strip_module_imports(path: pathlib.Path) -> str:
    """读取整个模块，去掉模块级 import（附录第 1 节已统一给出）。"""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, node.end_lineno + 1))
    seg = "".join(l for i, l in enumerate(lines, 1) if i not in drop)
    return re.sub(r"\n{4,}", "\n\n\n", seg).strip()


def build_prereq() -> tuple[str, int]:
    """导入 + 常量 + 数据结构 + 辅助函数，让附录成为可独立运行的脚本。"""
    all_files = [ROOT / rel for rel, _, _ in MODULE_SPEC] + [ROOT / "exp1_main.py"]
    imports = extract_imports(all_files)

    paths = [ROOT / rel for rel, _, _ in MODULE_SPEC]
    consts: list[tuple[str, str]] = []
    for p in paths:
        for nm in ("CLASS_NAMES", "CLASS_NAMES_CN"):
            consts += extract_assignments(p, [nm])

    dataclasses = (
        extract_defs(ROOT / "src/data_loader.py", ["Exp1Data"])
        + extract_defs(ROOT / "src/dim_reduction.py", ["PCAResult", "AEResult"])
        + extract_defs(ROOT / "src/feat_selection.py", ["SelectionResult"])
    )
    helpers = (
        extract_defs(ROOT / "src/data_loader.py", ["_load_label_table"])
        + extract_defs(ROOT / "src/dim_reduction.py", ["ae_reconstruct_error"])
        + extract_defs(ROOT / "src/feat_selection.py", ["_fit_l1_logistic"])
    )

    # 主程序里的模块级常量（去掉 PROJECT_ROOT / RANDOM_STATE，见下方统一声明）
    main_consts = extract_assignments(
        ROOT / "exp1_main.py",
        ["FIG_DIR", "TAB_DIR", "METRICS_PATH", "TOP_N_LIST", "PLOT_N", "METRICS"],
    )

    n_mod = 0
    parts = []
    parts.append("### 1. 导入依赖\n")
    parts.append("```python\n" + "\n".join(imports) + "\n```\n")

    parts.append("### 2. 路径与全局随机种子\n")
    parts.append(
        "> **注意**：本脚本需与 `实验1-数据/` 目录同级（即放在 `Exp1/` 下）才能找到数据文件。\n"
        "> 全部随机过程（t-SNE、自编码器、交叉验证划分）都固定为同一个种子，保证结果可复现。\n"
    )
    parts.append("```python\n"
                 "# 项目根目录：本脚本所在目录\n"
                 "PROJECT_ROOT = Path(__file__).resolve().parent\n"
                 "DATA_DIR = PROJECT_ROOT / \"实验1-数据\"\n"
                 "MAT_PATH = DATA_DIR / \"PNAS_HCC.mat\"\n"
                 "XLSX_PATH = DATA_DIR / \"PNAS_id_inforamtion.xlsx\"\n"
                 "\n"
                 "# 全局随机种子\n"
                 "RANDOM_STATE = 42\n"
                 "\n"
                 "# 类别名称：索引即 y 的取值\n"
                 + "\n".join(s for _, s in consts) + "\n"
                 "```\n")

    parts.append("### 3. 输出目录与实验参数\n")
    parts.append("```python\n" + "\n".join(s for _, s in main_consts) + "\n```\n")

    parts.append("### 4. 数据结构\n")
    parts.append("各模块统一使用的 dataclass 容器。\n")
    for name, seg in dataclasses:
        seg, k = strip_relative_imports(seg)
        n_mod += k
        parts.append(f"**`{name}`**\n")
        parts.append(code_block(seg))
    parts.append("")

    parts.append("### 5. 辅助函数\n")
    for name, seg in helpers:
        seg, k = strip_relative_imports(seg)
        n_mod += k
        parts.append(f"**`{name}`**\n")
        parts.append(code_block(seg))
    parts.append("")

    return "\n".join(parts), n_mod


def build_modules() -> tuple[str, int]:
    parts, n_mod = [], 0
    for rel, title, names in MODULE_SPEC:
        path = ROOT / rel
        parts.append(f"### {rel}\n")
        parts.append(INTRO[rel] + "\n")

        for name, seg in extract_defs(path, names):
            seg, k = strip_relative_imports(seg)
            n_mod += k
            parts.append(f"**`{name}`**\n")
            parts.append(code_block(seg))
        parts.append("")
    return "\n".join(parts), n_mod


def build_aliases() -> str:
    """模块别名：让 exp1_main.py 里的 dr.xxx / ev.xxx / fs.xxx / vz.xxx 原样可跑。

    原项目里主程序通过 `from src import dim_reduction as dr` 等方式访问各模块。
    拼成单文件后所有函数都在同一命名空间，用一个自引用即等价——这样主程序
    一个字符都不用改。
    """
    return (
        "### 模块别名（使主程序无需改动即可运行）\n\n"
        "原项目中主程序用 `dr.` / `ev.` / `fs.` / `vz.` 访问各模块。\n"
        "拼成单文件后所有函数都在同一命名空间，用下面的自引用即完全等价。\n\n"
        "```python\n"
        "import sys as _sys\n"
        "dr = ev = fs = vz = _sys.modules[__name__]\n"
        "```\n"
    )


def build_main_section() -> tuple[str, int]:
    """主程序（全量），体现完整处理流程。"""
    src = (ROOT / "exp1_main.py").read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    drop: set[int] = set()

    def drop_with_leading_comments(a: int, b: int):
        """连同紧邻在上方的注释行一起删除，避免留下孤立的注释。"""
        drop.update(range(a, b + 1))
        j = a - 1
        while j >= 1 and (lines[j - 1].lstrip().startswith("#")
                          or not lines[j - 1].strip()):
            if not lines[j - 1].strip():
                break
            drop.add(j)
            j -= 1

    for node in tree.body:
        # 去掉所有模块级 import（附录第 1 节已统一给出），以及 src.* 自引用、
        # sys.path 调整、重复的 PROJECT_ROOT
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop_with_leading_comments(node.lineno, node.end_lineno)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Attribute) and f.attr == "insert":
                drop_with_leading_comments(node.lineno, node.end_lineno)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PROJECT_ROOT":
                    drop_with_leading_comments(node.lineno, node.end_lineno)

    kept = [ln for i, ln in enumerate(lines, 1) if i not in drop]
    body = "".join(kept)
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip() + "\n"

    return (
        "### exp1_main.py（主程序，完整）\n\n"
        "这是完整处理流程的入口：加载数据 → 实验内容2 → 实验内容3 → "
        "实验内容4 → 汇总对比 → 写出 `results/`。\n"
        "（导入语句已在第 1 节统一给出，此处未重复；同时去掉了 `from src import ...`\n"
        "与 `sys.path` 调整——拼成单文件后这些名字已在同一文件顶层定义。）\n\n"
        + code_block(body)
        + "\n"
    ), len(drop)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def main():
    body = BODY.read_text(encoding="utf-8")
    if MARKER not in body:
        raise SystemExit(f"正文中缺少占位符 {MARKER}")

    prereq, n1 = build_prereq()
    modules, n2 = build_modules()
    aliases = build_aliases()
    main_sec, _ = build_main_section()

    appendix = (
        "## 附录：核心代码\n\n"
        "以下代码由 `report/gen_brief.py` 从 `src/` 与 `exp1_main.py` 的真实源文件中\n"
        "按符号名自动抽取，与跑出本报告结果的代码一致。抽取时只做了一处改动：\n"
        "把函数体内部的相对导入（`from .dim_reduction import ...`）去掉，"
        f"共 {n1 + n2} 处——拼成单文件后这些名字已在同一文件顶层定义。\n\n"
        "**关于完整性**：附录收录了全部算法与评估代码。绘图代码（`src/visualize.py`，\n"
        "约 250 行，仅含 `matplotlib` 绘图封装与中文字体配置）不属于算法核心，\n"
        "为控制篇幅未列入；补上该文件后，附录代码即可端到端运行并复现全部结果\n"
        "（已实测：拼装后 113 个指标字段与报告完全一致）。\n"
        "完整源码见提交的 `Exp1/` 目录，共 6 个模块 1801 行。\n\n"
        "### 源码结构\n\n"
        "```\n"
        "Exp1/\n"
        "├── exp1_main.py              主程序：一键完成实验内容 2/3/4\n"
        "├── src/\n"
        "│   ├── data_loader.py        数据加载与标签校验\n"
        "│   ├── dim_reduction.py      PCA / t-SNE / 自编码器\n"
        "│   ├── feat_selection.py     方差、熵、t 检验、MW、互信息、mRMR、Lasso\n"
        "│   ├── evaluate.py           轮廓系数、kNN 交叉验证、留出集重构误差\n"
        "│   └── visualize.py          绘图（中文字体配置）\n"
        "├── results/                  17 张结果图 + 4 张指标表 + metrics.json\n"
        "└── report/                   实验报告\n"
        "```\n\n"
        "---\n\n"
        "## 一、运行环境准备\n\n"
        + prereq
        + "\n---\n\n## 二、各模块核心代码\n\n"
        + modules
        + "\n"
        + aliases
        + "\n---\n\n## 三、主程序\n\n"
        + main_sec
    )

    OUT.write_text(body.replace(MARKER, appendix), encoding="utf-8")
    print(f"已生成 {OUT.name}")

    # 可运行副本 = 附录代码 + 绘图模块（绘图未列入附录，但运行必需）
    code_only = "\n\n".join(
        b.rstrip() for b in re.findall(r"```python\n(.*?)```", appendix, re.S)
    )
    viz = "\n\n".join(strip_module_imports(ROOT / f) for f in VERIFY_ONLY_FILES)

    # 绘图模块的 import 只注入到可运行副本，不写进报告附录
    # （附录里没有 matplotlib 代码，列出来会让人困惑）
    viz_imports = extract_imports([ROOT / f for f in VERIFY_ONLY_FILES])
    lines = code_only.splitlines()
    ins = next((i for i, l in enumerate(lines)
                if l.startswith("from __future__")), -1)
    lines[ins + 1:ins + 1] = [""] + viz_imports

    # 把 `if __name__ == "__main__": main()` 挪到文件最末尾——
    # 否则它会在追加的绘图模块之前执行，vz.plot_* 尚未定义。
    entry = [i for i, l in enumerate(lines) if l.startswith('if __name__')]
    entry_block: list[str] = []
    if entry:
        i = entry[-1]
        entry_block = lines[i:]
        del lines[i:]

    code_run = "\n".join(lines).rstrip()

    standalone = ROOT / STANDALONE_NAME
    standalone.write_text(
        "# 本文件 = 报告附录代码 + src/visualize.py（绘图部分，未列入附录）\n"
        "# 用途：验证附录代码可端到端复现报告中的全部结果。\n"
        "# 需与 实验1-数据/ 同级（即放在 Exp1/ 下）运行：python " + STANDALONE_NAME + "\n\n"
        + code_run + "\n\n"
        "# ---------------- 以下为 src/visualize.py（未列入报告附录）----------------\n\n"
        + viz + "\n\n"
        + "\n".join(entry_block) + "\n",
        encoding="utf-8",
    )
    print(f"已生成可运行副本 {STANDALONE_NAME}"
          f"（附录 {len(code_only.splitlines())} 行 + 绘图 {len(viz.splitlines())} 行）")

    if "--check" in sys.argv:
        run_check(standalone)


def run_check(standalone: pathlib.Path):
    print("\n" + "=" * 66)
    print("可复现性自检")
    print("=" * 66)

    # 1) 静态检查：是否还有未定义的名字
    r = subprocess.run([sys.executable, "-m", "pyflakes", str(standalone)],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    undef = [l for l in (r.stdout or "").splitlines() if "undefined name" in l]
    print(f"[1] pyflakes 未定义名字: {len(undef)} 条")
    for l in undef[:10]:
        print("     ", l)
    if undef:
        print("     -> 附录无法独立运行")
        return

    # 2) 实际执行，与参考结果比对
    import json
    ref_path = ROOT / "results" / "metrics.json"
    ref = json.load(open(ref_path, encoding="utf-8"))

    backup = ROOT / "results" / "_ref_backup.json"
    backup.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[2] 执行 {standalone.name}（约 1 分钟）...")
    r = subprocess.run([sys.executable, standalone.name], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(f"     执行失败 (returncode={r.returncode})")
        print("     " + (r.stderr or "")[-1500:])
        return
    print("     执行成功")

    # 3) 逐字段比对
    new = json.load(open(ref_path, encoding="utf-8"))

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{pre}{k}."))
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                out[pre + k] = tuple(round(x, 10) for x in v)
            else:
                out[pre + k] = v
        return out

    f1, f2 = flat(ref), flat(new)
    diffs = [(k, f1.get(k), f2.get(k)) for k in set(f1) | set(f2)
             if f1.get(k) != f2.get(k)]
    print(f"[3] 与报告结果逐字段比对（{len(f1)} 个字段）: 差异 {len(diffs)} 处")
    for k, a, b in diffs[:10]:
        print(f"      {k}: 报告={a}  重跑={b}")

    # 恢复参考结果（自检不应污染 results/）
    ref_path.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
    backup.unlink(missing_ok=True)

    print()
    print("  结论: 附录代码（补上未列入附录的绘图模块后）可端到端运行，"
          "且结果与报告完全一致"
          if not diffs else "  结论: 存在差异，需要排查")
    print("=" * 66)


if __name__ == "__main__":
    main()
