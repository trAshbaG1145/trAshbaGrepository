# Cancer-Alterome 查询数据的词汇计算实测报告

> **对照课件**：Lexicon Computing —— 基于词汇与计数的自然语言处理表示（夏静波，2026-09-17）
> **数据文件**：`CancerAlterome-query-download.tsv`
> **分析脚本**：`analyze_canceralterome.py`、`analyze_pmi_alias.py`
> **运行环境**：Python 3 + pandas 2.3.3 + scikit-learn 1.7.0
> **随机种子**：`TruncatedSVD(random_state=0)`

---

## 1. 任务与数据

### 1.1 数据来源与性质

本文件是 **Cancer-Alterome** 数据库的一次查询结果下载，四个字段：

| 字段 | 含义 | 取值情况 |
|---|---|---|
| `Cancer` | 癌种（查询条件） | 23 种 |
| `Gene` | 基因（查询条件） | **仅 1 个：ERBB2** |
| `PMID` | 文献编号 | 266 个唯一值 |
| `Sentence` | 从摘要中抽取的句子 | 315 条 |

对应课件 Team work 的文献：Xinzhi Yao, et al. *Cancer-Alterome: a literature-mined resource for regulatory events caused by genetic alterations in cancer.* Scientific Data. 2024, 11:265.

### 1.2 分析单位的确定（课件 1.1 节五要素）

| 要素 | 本项目的选择 |
|---|---|
| **对象** | 单条 `Sentence` 为最小单位；分析时按 `Cancer` 聚合成"文档" |
| **目标** | 描述性 + 探索性：在"已提及 ERBB2"的前提下，刻画各癌种的表述差异 |
| **输出** | 关键词表、关联强度排序、候选关系对、低维语义坐标 |
| **基线** | 词频 / TF-IDF（课件 4.2 节规定的最简基线） |
| **指标** | ① 稳定性（换写法结果是否漂移）② 人工可判读率 |

> **为何不做分类任务**：`Cancer` 是查询条件带来的元数据，不是从文本推断出的标签。以它为分类目标属于循环论证（详见第 5 节）。

---

## 2. 方法–课件对应总表

| 课件章节 | 方法 | 本数据中的实现 | 可行性 |
|---|---|---|---|
| 1.3 | Corpus / Document / Token / Type | 按 `Cancer` 切分文档，正则分词 | ✅ 直接可用 |
| 2.1 | TTR / MATTR | 按癌种聚合后计算 | ⚠️ 可用但**必须做长度控制** |
| 1.4 / 2.3 | 别名归一化 | 折叠 ERBB2 的 6 种写法 | ✅ **本数据最有价值的环节** |
| 3.2 | TF-IDF term-document matrix | 癌种为"主题 d"，词为"W" | ✅ 直接可用 |
| 3.3 | 词–词 PMI | 句子级共现估计 | ✅ 可用（窗口需报告） |
| 3.4 | SVD → LSA | TF-IDF 矩阵降维到 5 维 | ✅ 可用 |
| 3.5 | 误差分析 | 泄漏 / 重复 / 指代 / 偏差检查 | ✅ 发现 5 类问题 |
| 2.3 | NER 监督训练 | — | ❌ **无实体跨度标注** |
| 2.3 | 关系/事件抽取（有监督） | — | ❌ **无关系标签** |

---

## 3. 逐层实测结果

### 3.1 第一层：语料层级与统计量

```
行数（sentence 记录数）: 315
唯一 PMID             : 266
唯一 Cancer 类型       : 23
唯一 Gene             : 1  -> ['ERBB2']
```

**Token 统计（归一化前）**：

| 指标 | 值 |
|---|---|
| token 总数 N | 5524 |
| type 总数 V | 1493 |
| TTR = V/N | 0.2703 |
| 词频第 1 位 | `mutations` |

---

### 3.2 第二层：TTR / MATTR（课件 2.1 节）

按癌种分别计算：

| 癌种 | n_sent | N_tokens | V_types | TTR | MATTR(w=50) |
|---|---|---|---|---|---|
| Lung Adenocarcinoma | 140 | 3768 | 857 | 0.227 | 0.784 |
| Gastric Adenocarcinoma | 30 | 741 | 280 | 0.378 | 0.769 |
| Breast Ductal Carcinoma | 27 | 714 | 293 | 0.410 | 0.790 |
| Hepatocellular Carcinoma | 13 | 394 | 220 | 0.558 | 0.823 |
| Esophageal Carcinoma | 14 | 382 | 194 | 0.508 | 0.768 |
| Pancreatic Ductal Adenocarcinoma | 12 | 354 | 207 | 0.585 | 0.854 |
| Cholangiocarcinoma | 11 | 335 | 186 | 0.555 | 0.805 |
| Head and Neck Squamous Cell Carcinoma | 10 | 286 | 164 | 0.573 | 0.812 |
| Colorectal Adenocarcinoma | 7 | 225 | 140 | 0.622 | 0.823 |
| Prostate Adenocarcinoma | 7 | 216 | 121 | 0.560 | 0.740 |
| Glioblastoma Multiforme | 8 | 208 | 125 | 0.601 | 0.832 |
| Cervical Carcinoma | 7 | 202 | 129 | 0.639 | 0.805 |
| Lung Squamous Cell Carcinoma | 5 | 160 | 93 | 0.581 | 0.791 |
| Uterine Carcinosarcoma | 3 | 126 | 90 | 0.714 | 0.791 |
| Breast Lobular Carcinoma | 7 | 120 | 81 | 0.675 | 0.801 |
| Sarcoma | 4 | 105 | 67 | 0.638 | 0.742 |
| Bladder Urothelial Carcinoma | 3 | 73 | 53 | 0.726 | 0.768 |
| Kidney Clear Cell Carcinoma | 2 | 55 | 46 | 0.836 | 0.867 |
| Kidney Chromophobe Carcinoma | 1 | 43 | 31 | 0.721 | — |
| Acute Myeloid Leukemia | 1 | 29 | 28 | **0.966** | — |
| Mesothelioma | 1 | 27 | 23 | 0.852 | — |
| Ovarian Serous Adenocarcinoma | 1 | 25 | 20 | 0.800 | — |
| Testicular Germ Cell Cancer | 1 | 22 | 21 | 0.955 | — |

#### 关键验证：课件的"长度效应"警告成立

```
corr(N_tokens, 原始 TTR)   = -0.776     ← 长度效应极强
corr(N_tokens, MATTR_w50)  = -0.122     ← 长度效应被削弱
```

- **原始 TTR 与文本长度强负相关（−0.776）**，与课件 2.1 节"随着文本变长，新的 token 更容易重复已有词型，因此 V/N 往往下降"完全一致。
- **MATTR 把相关性压到 −0.122**，验证了课件给出的改进方案有效。
- **反例**：Acute Myeloid Leukemia 只有 1 句话、29 个 token，TTR 高达 0.966 位列第一。这个数字**没有任何语言多样性含义**，纯粹是短文本的数学后果。

> **结论**：本数据的 TTR 在癌种间**不可直接比较**。必须报告 N 与 MATTR 窗口长度，或只比较长度可比的癌种对。

---

### 3.3 第三层：别名归一化（课件 1.4 / 2.3 节）★核心发现

课件在讲 NER 难点时列出"缩写与别名"会导致错误。本数据是这条论断的现场证据。

#### 实测：ERBB2 的写法分裂

| 写法 | 出现次数 | 出现于句子数 |
|---|---|---|
| `HER2` | 231 | 186 |
| `ERBB2` | 131 | 116 |
| `neu` | 37 | 33 |
| `Her2/neu` | 26 | 23 |
| `c-erbB-2` | 22 | 22 |
| **合计** | **448** | — |

**同一个基因被拆成 5~6 个互不相干的 token。** 而 `Gene` 列只有一个标签 `ERBB2`——这正是课件 1.4 节说的"归一化规则会改变下游一切统计量"。

#### 实测：两种写法给出了两套不同的"关联词"

分别以 `erbb2` 和 `her2` 为锚点计算词–词 PMI：

| erbb2 的 top 伙伴词 | PMI | her2 的 top 伙伴词 | PMI |
|---|---|---|---|
| pathway | 9.24 | present | 8.97 |
| tp53 | 9.21 | nsclc | 8.75 |
| reported | 9.15 | kras | 8.33 |
| signaling | 8.94 | lung | 8.30 |
| tyrosine | 8.89 | reported | 8.22 |
| kinase | 8.83 | human | 8.10 |
| genes | 8.77 | higher | 7.92 |
| kras | 8.62 | patients | 7.92 |
| receptor | 8.58 | mutation | 7.91 |
| mutations | 8.02 | prognosis | 7.76 |

**两个列表的 top-30 只交集 13 个词：**

```
仅在 erbb2 侧: expression, gastric, gene, genes, growth, pathway, signaling, significantly
仅在 her2  侧: nsclc, overexpression, present, prognosis, protein, tumors
```

同一份数据、同一个基因，仅因写法不同，就得到"信号通路导向"与"临床预后导向"两套结论。

#### 实测：折叠后的统计量变化

| 指标 | 归一化前 | 归一化后 |
|---|---|---|
| token 总数 N | 5524 | 5542 |
| type 总数 V | 1493 | 1485 |
| TTR | 0.2703 | 0.2680 |
| `erbb2` 词频 | 120 | **378** |
| `her2` 词频 | 172 | —（已折叠） |
| ERBB2 词频排名 | 第 4 位 | **第 1 位** |

**折叠后 ERBB2 从第 4 位跃升到第 1 位。** 归一化后词频 Top-15：

```
erbb2norm 378 | mutations 155 | egfr 129 | amplification 106 | lung 86
cancer 78 | patients 77 | gene 73 | mutation 68 | associated 53
overexpression 48 | p 42 | receptor 40 | kinase 39 | tumor 39
```

注意 `mutations`(155) 与 `mutation`(68) 仍然分离——**词形（stemming/lemmatization）是课件 1.4 节列出的另一项研究决策，本项目尚未处理**。

---

### 3.4 第四层：TF-IDF（课件 3.2 节）

**term-document matrix 形状：`(23, 1447)`**（23 个癌种 × 1447 个词）

参数：`stop_words='english'`, `sublinear_tf=True`, `token_pattern=r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b"`

#### 每个癌种 TF-IDF 权重最高的 6 个词

| 癌种 | Top 关键词 | 可判读性 |
|---|---|---|
| Breast Ductal Carcinoma | dcis, ci, hypomethylation, subtype, high-grade, ductal | ✅ |
| Cholangiocarcinoma | cholangiocarcinoma, fusions, fgfr2, hepatocyte, aberrant, idh1 | ✅ |
| Colorectal Adenocarcinoma | msi-h, pd-1, mutated, lower, mutation, human | ✅ |
| Pancreatic Ductal Adenocarcinoma | pancreatic, ductal, ink4a, dpc4, activating, k-ras | ✅ |
| Lung Adenocarcinoma | alk, lung, exon, nsclc, ret, phosphorylation | ✅ |
| Gastric Adenocarcinoma | gastric, loh, differentiated, gc, aim, k-sam | ✅ |
| Hepatocellular Carcinoma | hcc, postoperative, level, afp, hbv, staging | ✅ |
| Prostate Adenocarcinoma | prostate, colon, androgen, breast, important, cancer | ⚠️ 混入其他癌种词 |
| Mesothelioma | detected, various, non-small, lines, cell, braf | ❌ 仅 1 句，无意义 |

#### 复现课件的课堂思考题

课件原题是"计算 tf-idf('的', 癌症) 与 tf-idf('的', 心理学)"——考察同一个词在不同主题下的权重差异。本数据的同构版本：

| 词 | TF-IDF 最高的癌种 |
|---|---|
| `dna` | Colorectal Adenocarcinoma = 0.098，Lung Adenocarcinoma = 0.055 |
| `her2` | Bladder Urothelial Carcinoma = 0.168，Breast Lobular Carcinoma = 0.154 |
| `amplification` | Ovarian Serous Adenocarcinoma = 0.164，Bladder Urothelial Carcinoma = 0.141 |
| `prognosis` | Kidney Clear Cell Carcinoma = 0.109，Prostate Adenocarcinoma = 0.093 |
| `mutation` | Breast Lobular Carcinoma = 0.155，Colorectal Adenocarcinoma = 0.152 |

与课件示例同理：`dna` 在结直肠癌里权重高，是**该主题的特征**；若把 `dna` 当作通用词删掉，这类信号就丢失了——对应课件说的"停用词删除后，领域高频词也被误删"。

---

### 3.5 第五层：词–词 PMI（课件 3.3 节）

**实现说明**：课件只给了公式 $\text{PMI}(w^i,w^j)=\log_2\frac{P(w^i,w^j)}{P(w^i)P(w^j)}$，未指定窗口。本报告采用：
- $P(w^i,w^j)$ = 两词**共现句子数 / 总句数**
- $P(w^i)$, $P(w^j)$ = **token 频率**
- 共现阈值 `min_cooccur = 8` 句

以 `ERBB2` 为锚点的 top 伙伴词：`pathway(9.24), tp53(9.21), signaling(8.94), tyrosine(8.89), kinase(8.83), kras(8.62), receptor(8.58)`。

**生物学可判读性**：ERBB2 是受体酪氨酸激酶（receptor tyrosine kinase），其 top 关联词恰好是 `receptor / tyrosine / kinase / signaling / pathway`，与 `kras / tp53` 的共现也符合已知的 MAPK 与 p53 通路交互。**方法在这份数据上产出了生物学上说得通的结果。**

> ⚠️ 换成 ±5 词窗口或 PMI^k 平滑，排序会变化。**窗口长度必须随结果一起报告**（课件 3.5 节"解释时必须同时报告"）。

---

### 3.6 第六层：SVD → LSA（课件 3.4 节）

```
原始 TF-IDF 矩阵稀疏度: 91.63%     ← 课件 "The representation is always sparse"
LSA 降维后形状: (23, 5)  (稠密)
各奇异值解释方差比: [0.017 0.059 0.058 0.055 0.053]
累计解释方差比    : 0.242
```

稀疏度 91.63% 印证了课件的论断；5 维稠密表示累计解释 24.2% 方差。

#### 癌种在 LSA 前两维的坐标（课件散点图的对应物）

| 癌种 | dim1 | dim2 |
|---|---|---|
| Lung Adenocarcinoma | +0.602 | −0.199 |
| Gastric Adenocarcinoma | +0.576 | +0.030 |
| Colorectal Adenocarcinoma | +0.485 | +0.019 |
| Esophageal Carcinoma | +0.483 | +0.083 |
| Breast Ductal Carcinoma | +0.457 | +0.187 |
| Kidney Chromophobe Carcinoma | +0.213 | **+0.404** |
| Ovarian Serous Adenocarcinoma | +0.325 | **+0.515** |
| Lung Squamous Cell Carcinoma | +0.346 | **−0.447** |
| Mesothelioma | +0.291 | −0.437 |

`Lung Squamous Cell Carcinoma` 与 `Mesothelioma` 在 dim2 上相邻（−0.447 / −0.437），二者确实都是胸部肿瘤——**无监督降维恢复出了部分解剖学语义结构**。

#### 词在 LSA 空间中的近邻（回答课件的提问）

课件问："经矩阵分解，获得的列向量和行向量分别能指代什么信息？"本数据的实测回答：

| 锚点词 | LSA 空间近邻 |
|---|---|
| `her2` | common, proto-oncogene, kras, status, efficacy |
| `amplification` | case, overexpression, cancer, type, features |
| `prognosis` | patients, younger, mutation, histology, mainly |
| `trastuzumab` | differentiated, background, locus, loh, heterozygosity |

- **行向量（$U$ 的行）** → 癌种在潜在语义空间的坐标，可做主题聚类
- **列向量（$V^T$ 的列）** → 词在潜在语义空间的坐标，可做近义词发现

`trastuzumab` 的近邻质量明显较差（出现 loh、heterozygosity 等无关词），说明 **5 维对这部分语义不够**，且该词样本量小。

---

## 4. 误差分析（课件 3.5 节要求）

### 4.1 词汇层面的错误

| 问题 | 实测证据 |
|---|---|
| 同一概念被大小写/连字符/缩写分开 | ERBB2 分裂为 6 种写法，导致 PMI 结果漂移（见 3.3） |
| 词形未归一 | `mutations`(155) 与 `mutation`(68) 分别计数 |
| 领域高频词可能被误删 | `dna`、`her2` 等是强主题信号（见 3.4） |
| 指代无法解析 | **72/315（22.9%）** 句子以 These/This/However/Here/Thus 等开头 |

指代问题的实例：`"These results suggest that amplification and overexpression of c-erbB-2 may be correlated with metastases..."` —— 脱离摘要上下文后，"These results"指向的发现无从确定。这是**句子级抽取的固有限制**，不是代码 bug。

### 4.2 实验层面的错误

| 问题 | 实测证据 | 后果 |
|---|---|---|
| **数据泄漏（同一文献跨类别）** | **9 个 PMID 出现在多个 `Cancer` 下**；PMID 16397024 横跨 3 个癌种 | 若按 Cancer 做分类，同篇论文会同时落进训练集和测试集 |
| **完全重复行** | 10 行 `(PMID, Sentence)` 完全重复 | 重复计数会放大某些词的权重 |
| **类别极度不平衡** | Lung Adenocarcinoma 140 条 vs 8 个癌种各 1 条 | 任何跨组比较都被最长的那组主导 |
| **选择偏差** | 全部 315 条都已保证提及 ERBB2 | 见 4.3 |

### 4.3 选择偏差（本数据最致命的问题）

**本文件是按 `Gene = ERBB2` 过滤出的查询结果，不是随机文献样本。**

因此以下结论**全部是循环论证**：

- ❌ "ERBB2 在癌症文献中高频出现" —— 由查询条件定义
- ❌ "ERBB2 与 kinase 强关联" —— 查询只返回了提到 ERBB2 的句子
- ❌ "ERBB2 在肺癌中最重要" —— 140/315 是肺癌，由数据库覆盖度决定

**可以做的**是**条件分布分析**：在"已知提及 ERBB2"的前提下，不同癌种的**表述方式**有何差异（如胃癌侧多说 `amplification / trastuzumab`，肺癌侧多说 `exon 20 / nsclc`）。这个 conditioning 必须写在结论里。

同样地，`Cancer` 标签来自查询元数据，**不能当作分类目标**——用它训练分类器等于用答案预测答案。

---

## 5. 结论

### 5.1 可以做的

| 课件方法 | 应用 |
|---|---|
| BoW / 词频 | 全语料 token/type 统计、高频词表 |
| TTR / MATTR | 各癌种的**长度可控**词汇多样性比较 |
| 别名归一化 | 合并 ERBB2 写法，稳定下游统计 |
| **TF-IDF** | 各癌种的关键词提取（结果可判读） |
| **PMI** | 候选关系对生成（ERBB2 的关联词生物学可解释） |
| **SVD → LSA** | 癌种与词的潜在语义坐标、近邻发现 |

**这些精确对应课件 4.2 节表中所列的"第一轮可采用的方法：TF-IDF 初筛；词典/规则识别实体；共现/PMI 生成候选关系"。**

### 5.2 不能做的

| 课件任务 | 原因 |
|---|---|
| 有监督 NER | `Gene` 列只有 1 个取值，**无实体跨度标注**，无监督信号 |
| 关系/事件抽取（有监督） | **无关系类型标签**，无三元组标注 |
| 主题分类（以 Cancer 为目标） | 标签来自查询条件，属循环论证 |
| ERBB2 的文献计量结论 | 选择偏差，见 4.3 |

### 5.3 一句话总结

**这份数据能做课件说的"第一轮基线"——TF-IDF 关键词、PMI 候选关系、LSA 语义坐标都跑通且结果可解释；但它是一份"已知提及 ERBB2"的条件样本，任何关于 ERBB2 本身频次/重要性的推断都是循环的，且别名不归一化会让结论随写法漂移。**

---

## 6. 最小实验记录（课件 3.5 节要求）

```
question.md           # 研究问题：在已知提及 ERBB2 的句子中，
                      #   不同癌种的表述与关联词有何差异？
                      # 预期方向：肺癌侧偏 exon/insertion，胃癌侧偏 amplification/trastuzumab
data_manifest.csv     # CancerAlterome-query-download.tsv，315 行，266 PMID，23 癌种
preprocess.py         # 正则分词 + ERBB2 别名折叠 + 英文停用词
run.sh                # python analyze_canceralterome.py && python analyze_pmi_alias.py
结果表                # 本报告第 3 节
失败样例              # Mesothelioma(1句)、Ovarian(1句) 等 8 个单句癌种的 TF-IDF/LSA 均不可用
结论                  # 本报告第 5 节
```

### 已固定的预处理规则（课件 1.5 节要求）

| 决策项 | 本报告的选择 |
|---|---|
| 分词 | `re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)` |
| 大小写 | 全部转小写 |
| 停用词 | 英文通用停用词表（约 100 词，见脚本 `STOP`） |
| ERBB2 别名 | `\b(ERBB2\|HER-?2\|Her2/neu\|c-erbB-?2)\b` → `ERBB2NORM` |
| 词形归一 | **未做**（`mutations`/`mutation` 仍分离） |
| 数字与标点 | 保留数字（统计量密集句 51 条），标点丢弃 |
| 文档边界 | 一个癌种 = 一个 document（聚合其全部句子） |

---

## 7. 下一步建议

按课件 3.5 节"提交结果前，按以下顺序组织实验"，尚缺的部分：

1. **词形归一化**：加 stemming 或 lemmatization，消解 `mutations`/`mutation` 类分裂，观察 TF-IDF 结果如何变化 —— 这是一次现成的**消融实验**
2. **按 PMID 分组划分**：为任何未来的分类实验准备无泄漏的数据划分
3. **人工抽样核验**：从 23 个癌种各抽 5 条，判读 TF-IDF 关键词是否正确，记录可判读率
4. **换窗口重算 PMI**：用 ±5 词窗口替代句子级共现，对比候选关系表的变化
5. **对齐 Team work 问题**：这篇论文中哪些方法属于"基于词汇计数"、哪些属于"神经网络方法"（对应课件 4.3 节的四问）

---

## 附：脚本清单

| 文件 | 内容 |
|---|---|
| [`analyze_canceralterome.py`](analyze_canceralterome.py) | 语料统计、TTR/MATTR、别名统计、TF-IDF、词–主题 PMI、SVD/LSA、误差分析 |
| [`analyze_pmi_alias.py`](analyze_pmi_alias.py) | 词–词 PMI（双锚点对比）、别名归一化前后对比 |
| [`Lexicon_Computing_讲义笔记.md`](Lexicon_Computing_讲义笔记.md) | 课件完整笔记 |
