# PubMed 中的生物实体识别（NER）、Shell 编程与 NLTK 实现

> 课程：生物文本挖掘与知识发现（本硕贯通）
> 讲者：Jingbo Xia（华中农业大学，xiajingbo.math@gmail.com）
> 日期：2026-10-08
> 来源：`JingboSlides_8.PubMedTermsNERandShellProgramming.pdf`（63 页）

---

## 工具地图与本讲叙事线（p2–3）

**Highlight**：PubTator 由 NCBI 团队持续维护，是对 PubMed 生物实体进行自动标注的数据库——PubMed 摘要数据与 PMC 全文均已获得广泛实体标注，主要覆盖 **Gene、Chemical、Species、Mutation、Disease** 等常用类别。获取海量 PubTator 标注文件并非难事，**知识的进一步挖掘和加工**才是主要的考验。

|      | PubTator                      | NLTK                         |
| ---- | ----------------------------- | ---------------------------- |
| 定位 | 生物文献标注服务与数据库      | 通用 NLP 教学库              |
| 知识 | 内置 BioNER 模型 + 标准标识符 | 无领域知识                   |
| 输出 | 实体提及 + 标准编号           | 句子、词元、词频             |
| 运行 | 远程 API，超模型版本化        | 本地运行，版本可控           |
| 分工 | **提供标注（原材料）**  | **负责加工（结构化）** |

> 共性：都是"文本 → 结构化信息"的转换器；输出均可复核（字符偏移 / 句子编号）；都可编程访问（RESTful API / Python 库）；都**不直接产出生物学结论**。

**叙事线**：PubTator 获取标注 → Shell 批量下载 → RiceMind 对照（已结构化证据）→ NLTK 加工为句子级证据表。

### 目录

1. PubMed（p5）
2. PubTator（p24）
3. RiceMind 数据与 API 访问（p32）
4. 代码实验：NLTK 实现文本预处理（p38）
   - 数据结构与环境准备（p39）
   - NLTK 基础实现（p45）
   - 句子级证据构建（p50）
   - 验证与课堂练习（p54）
5. 讨论：面向水稻数据的自主实体识别开发（p57）

---

## 1. PubMed（p5–23）

- PubMed 收录 **3300 万+** 生物医学文献引用（来自 MEDLINE、生命科学期刊、在线图书），可链接到 PMC 和出版商网站的全文。
- 主要页面：[https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/)
- 参考材料：「Welcome to E-utilities for PubMed」视频、DataGuid「How does E-utilities work?」页。

### E-utilities API 要点

**esearch**（检索，得到 pmid 列表）：

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=nature[journal]+AND+3D+printing
```

返回 XML 关键字段：

```xml
<eSearchResult>
  <Count>51</Count>
  <RetMax>20</RetMax>          <!-- 默认只返回 20 条 -->
  <RetStart>0</RetStart>
  <IdList>
    <Id>34408301</Id>
    ...
  </IdList>
</eSearchResult>
```

- 课堂思考：**默认下载 20 个 pmid**，如何改？
- MeSH Terms 检索 vs All Fields 检索的命中量差异（`printing[All Fields]` 30858 vs MeSH 9433 等）；`Translation` 项有助于文献细分。
- 加 `&retmax=50&usehistory=y` 后返回 `<WebEnv>MCID_...` 与 `<QueryKey>1</QueryKey>`——**usehistory=y 用于获取 WebEnv**，供后续 efetch 引用同一检索结果集。
- 课堂思考：要下载 50 篇 Covid19 文献？

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=covid19[mesh]&retmax=50&usehistory=y
```

**efetch**（按检索集/按 pmid 取正文）：

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&query_key=1&WebEnv=MCID_...&retstart=0&retmax=50&retmode=text&rettype=abstract
```

- 课堂思考：下载**所有**有关 Covid19 的文献，`retmax=?` → 答：**9999 restriction applied**（当年示例数值 150159，实际随时间变化）。
- 取全部 pmid 列表：`...esearch.fcgi?db=pubmed&term=covid19[mesh]&retmax=150159`
- Shell 清洗（逐级管道）：

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=covid19[mesh]&retmax=150159" > pmid_covid19.txt

cat pmid_covid19.txt | grep "<Id>"
cat pmid_covid19.txt | grep "<Id>" | sed 's/<Id>//g'
cat pmid_covid19.txt | grep "<Id>" | sed 's/<Id>//g' | sed 's/<\/Id>//g'
cat pmid_covid19.txt | grep "<Id>" | sed 's/<Id>//g' | sed 's/<\/Id>//g' | sed 's/[[:blank:]]//g'
```

- 课堂思考：依据 pmid 获得论文摘要？

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=39371582&retmode=xml
```

---

## 2. PubTator（p24–31）

- PubTator 是面向文献人工策展（标注生物实体及其关系）的 Web 工具，基于先进文本挖掘技术，提供标注 PubMed 引用的**一站式服务**；新版 PubTator Central（2.0）自 2020 年起作为默认。
- 主页：[https://www.ncbi.nlm.nih.gov/research/pubtator/](https://www.ncbi.nlm.nih.gov/research/pubtator/)

### RESTful API

API 格式（PubTator / JSON / BioC）：

```
https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/[Format]?[Type]=[Identifiers]&concepts=[Bioconcepts]
```

示例（PubTator 格式，返回带实体标注的文献）：

```
https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids=19894120
```

返回示例（节选）：

```
19894120|t|Lipoproteins increase the expression of multidrug resistance-associated protein 1 (MRP1) ...
19894120|a|Multidrug resistance-associated protein 1 (MRP1) is a ubiquitously expressed ...
19894120	47	58	lipoproteins	Chemical	MESH:D008080
19894120	99	108	MRP1	Gene	4363
19894120	205	265	Multidrug resistance-associated protein 1	Gene	4363
...
```

示例（BioC-XML，限定 gene 概念）：

```
https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids=19894120&concepts=gene
```

### Shell programming（批量下载）

第一版脚本：

```bash
#!/bin/bash
F_OUT="tmp.txt"

echo -e "\nLet us curate the results!\n"

curl https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids=19894122 >$F_OUT
```

```bash
$ chmod 755 NCBI1.sh   # if you do not have permission to run the shell
```

> Note：`F_OUT` 变量值为 "tmp.txt"；`echo -e` 的反斜杠转义选项使 `\n` 被识别为换行符。

批量循环版：

```bash
#!/bin/bash
F_OUT="result_PubTator.txt"
F_LIST="pmid_list.txt"

echo -e "\n Let us curate the results!\n"
echo -e "\n" >$F_OUT

total=$(wc -l < "$F_LIST")
i=1
while IFS= read -r line
do
  print "..."   # 见下一版
done <"$F_LIST"
```

完整版（按 pmid 列表逐条请求 + 限速）：

```bash
#!/bin/bash
F_OUT="result_PubTator.txt"
F_LIST="pmid_list.txt"

echo -e "\n Let us curate the results!\n"
echo -e "\n" >$F_OUT

total=$(wc -l < "$F_LIST")
i=1
while IFS= read -r line
do
  url="https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids=$line"
  curl "$url" >>$F_OUT
  print "..."   # 进度提示
  i=$((i+1))
  sleep 5.8s    # 避免请求过快
done <"$F_LIST"
```

> Note：`$` 变量名用于获取变量的值。

---

## 3. RiceMind 数据与 API 访问（p32–37）

RiceMind 面向水稻文献证据的查询与整理，将基因、性状、品种、PMID 以及句子证据组织为**可追溯**的数据对象。本章目标不是"看一个数据库页面"，而是学会**提出查询、读取 JSON、保留证据来源，并用 Shell 完成可复现筛选**。

### 从文献到可计算证据（四层）

| 层     | 内容                                         |
| ------ | -------------------------------------------- |
| 文献层 | PMID、题名、期刊、年份、DOI                  |
| 证据层 | 句子文本（text）及其句子编号（sent_id）      |
| 实体层 | 基因（Gene）、性状（Trait）和品种（Variety） |
| 评价层 | 来源（sources）、证据代码和置信度层级        |

> 课堂思考：如果一个基因在很多句子中出现，是否就可以直接断言它造成了某个性状？为什么还要保留原始句子和证据代码？

### RiceMind API

基础地址：`http://lit-evi.hzau.edu.cn/ricemind-api/`

按基因查询：

```
Gene profile:
http://lit-evi.hzau.edu.cn/ricemind-api/gene-profile/?gene=sd1

基因相关的句子证据：
http://lit-evi.hzau.edu.cn/ricemind-api/search-by-gene/?gene=sd1&page=1&limit=20

基因关联的性状：
http://lit-evi.hzau.edu.cn/ricemind-api/traits-by-gene/?gene=sd1&confidence=All&onto_type=ALL&page=1&limit=20
```

按性状查询（性状名称应使用数据库标准标签，如 `drought tolerance`）：

```
http://lit-evi.hzau.edu.cn/ricemind-api/search-by-trait/?trait=drought%20tolerance&page=1&limit=20
```

按基因-性状组合查询：

```
http://lit-evi.hzau.edu.cn/ricemind-api/search-by-trait-and-gene/?gene=sd1&trait=drought%20tolerance&page=1&limit=20
```

> **解释边界**：API 返回的候选记录、置信度和证据代码用于**组织与筛选文献证据**，不等同于最终的生物学结论；句子中的共现关系也不自动等于因果关系。

### 读取 JSON 时应保留什么

- 分页信息：`current_page`、`total_pages`、`total_count`
- 文献定位：`PMID`、`year`、`journal`、`title`、`doi`
- 句子定位：`sent_id`、`text`
- 证据解释：`gene`、`trait`、`confidence_tier`、`evidence_codes`
- 可追溯性：原始 JSON 文件、查询参数和下载日期

> 课堂思考：为什么不能只保存"基因名 + 性状名"的二维表？请从**复核证据、去除误匹配和复现实验**三个角度讨论。

### Shell 示例

```bash
#!/bin/bash
BASE="http://lit-evi.hzau.edu.cn/ricemind-api"
GENE="sd1"

# 保存一个基因的概览信息
curl -fsSL "$BASE/gene-profile/?gene=$GENE" \
  -o "$GENE.profile.json"

# 保存该基因的第1页句子证据
curl -fsSL "$BASE/search-by-gene/?gene=$GENE&page=1&limit=20" \
  -o "$GENE.evidence.page1.json"
```

> 检查与讨论：
>
> - 检查返回结果是否为合法 JSON，找出 `total_pages` 并思考如何循环下载全部页面；
> - 保存原始响应并记录查询参数（基因、性状、page、limit），保证结果可复现。
>
> 课堂思考：能否仿照前面 PubTator 的代码：自动遍历所有页面，筛选包含目标性状的句子，并同时输出 PMID、sent_id、text 和原始 JSON？

---

## 4. 代码实验：NLTK 实现文本预处理（p38–56）

### 4.1 数据结构与环境准备（p39–44）

**本章基本问题**：如何把带有生物医学标注的文本，转换成可以检查、统计和复核的**句子级证据**？

```
PubTator 文本 → NLTK 预处理 → 句子级证据表
```

| INPUT                                                            | OUTPUT                                                    |
| ---------------------------------------------------------------- | --------------------------------------------------------- |
| 一篇或多篇摘要；原始文本；实体标注及其字符位置；关键词或实体类型 | 句子编号；句子文本；实体提及及其类型；PMID 和标准化标识符 |

> 课堂思考：每一步都要回答三个问题——**输入是什么？代码改变了什么？输出能否被别人复核？**

**PubTator 输入格式**：

```
20301443|t|Peutz-Jeghers syndrome
20301443|a|STK11 mutations cause this syndrome.

20301443<TAB>0<TAB>22<TAB>Peutz-Jeghers syndrome<TAB>Disease<TAB>MESH:D010580
20301443<TAB>23<TAB>28<TAB>STK11<TAB>Gene<TAB>6794
```

| 字段       | 含义                                                                                    |
| ---------- | --------------------------------------------------------------------------------------- |
| `          | t                                                                                       |
| `          | a                                                                                       |
| begin/end  | 字符偏移（偏移量基于"标题 + 空格 + 摘要"的拼接文本，从 0 开始，end 为**开区间**） |
| mention    | 文本中的提及                                                                            |
| type       | 实体类型                                                                                |
| identifier | 标准编号                                                                                |

> 验证：`text[23:28]` 是否等于 `"STK11"`。
> 教学重点：**NLTK 负责文本结构处理；PubTator 负责提供生物医学实体及其标准化标识。** `<TAB>` 表示原文件中的制表符。

**第一步：把 PubTator 文件读成结构化数据**

```python
from pathlib import Path

def parse_pubtator(path):
    pmid, title, abstract, entities = None, "", "", []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "|t|" in line:
            pmid, title = line.split("|t|", 1)
        elif "|a|" in line:
            pmid, abstract = line.split("|a|", 1)
        elif line.strip():
            f = line.split("\t")
            entities.append({"pmid": f[0], "begin": int(f[1]),
                             "end": int(f[2]), "mention": f[3],
                             "type": f[4], "identifier": f[5]})
    return {"text": title + " " + abstract, "entities": entities}
```

> **关键约定**：返回的 `text` 使用与偏移量一致的拼接规则（标题 + 空格 + 摘要），因此每个实体都应满足 `text[begin:end] == mention`，**这本身就是一条自动检查**。

**NLTK 在本章中做什么？**

| 适合完成       | 不能过度承诺             |
| -------------- | ------------------------ |
| 句子切分       | NLTK 不是生物医学知识库  |
| 词元化         | 通用 NER 不等于 BioNER   |
| 词频统计       | 词频不等于生物学重要性   |
| 基础文本清理   | 自动切分结果需要抽样检查 |
| 句子级结果组织 |                          |

> 本章原则：让工具处理重复性文本操作，让研究者保留数据定义、结果检查和科学解释。

**环境准备**：

```bash
pip install nltk
```

```python
import nltk
nltk.download("punkt")      # sent_tokenize / word_tokenize 所需
nltk.download("punkt_tab")  # NLTK >= 3.9 还需要此资源
```

> 常见报错：运行 `sent_tokenize` 时报 `LookupError: Resource punkt not found`，说明上述数据包尚未下载——这是本章实验最常见的环境问题。
> 课堂思考：为什么 NLTK 把"代码"和"语言模型数据"分开安装？这对结果的**可复现性**意味着什么？

### 4.2 NLTK 基础实现（p45–49）

**案例一：句子切分**

```python
from pathlib import Path
from nltk.tokenize import sent_tokenize

text = Path("abstract.txt").read_text(encoding="utf-8")
sentences = sent_tokenize(text)

for sent_id, sentence in enumerate(sentences, start=1):
    print(sent_id, sentence)
```

- 输出不是"若干行字符串"，而是一个**有顺序的句子列表**：`sent_id` 将成为后续证据定位的索引。

| INPUT                                                                                                           | OUTPUT                                                                                                       |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `P53 mutations were detected in several samples.` / `The mutations were associated with tumor progression.` | `sentence_id \| sentence`：`1 \| P53 mutations were detected...`、`2 \| The mutations were associated...` |

> 学生应能解释：`sent_tokenize` 把一个长字符串转换成句子序列；**它没有识别 Gene 或 Disease，只完成文本结构切分。**

**案例二：词元化与词频统计**

```python
from collections import Counter
from nltk.tokenize import word_tokenize

tokens = []
for sentence in sentences:
    tokens.extend(word_tokenize(sentence))

words = [
    token.lower()
    for token in tokens
    if token.isalpha()
]

frequency = Counter(words)
for word, count in frequency.most_common(10):
    print(word, count)
```

- 处理链：`sentence → tokens → 小写化 → 去除标点 → 词频表`
- 中间结果示例：`["P53", "mutations", "were", "detected", "in", "samples"]` → 清理后 `["mutations", "were", "detected", "samples"]`
- 输出示例：`mutations 12`、`detected 9`、`samples 7`

> **解释边界**：词频高不等于生物学重要；数字、基因符号、连字符和大小写处理都会改变统计结果。

### 4.3 句子级证据构建（p50–53）

**案例三：把实体标注对齐到句子**

```python
from nltk.tokenize import sent_tokenize

def sentence_spans(text):
    spans, cursor = [], 0
    for sent_id, sent in enumerate(sent_tokenize(text), start=1):
        start = text.find(sent, cursor)
        if start == -1:
            raise ValueError("sentence alignment failed")
        end = start + len(sent)
        spans.append({"sent_id": sent_id, "begin": start,
                      "end": end, "text": sent})
        cursor = end
    return spans
```

- **为什么要保留字符位置**：PubTator 实体标注使用字符偏移；先定位句子的 `begin/end`，才能判断实体属于哪个句子。
- **这段代码的脆弱性**：`text.find` 要求切分结果与原文**逐字符一致**；空白字符被改动则对齐失败。

输出示例（句子级证据表）：

| PMID     | sentence_id | mention                | type    | identifier   |
| -------- | ----------- | ---------------------- | ------- | ------------ |
| 20301443 | 1           | Peutz-Jeghers syndrome | Disease | MESH:D010580 |
| 20301443 | 1           | STK11                  | Gene    | 6794         |

> OUTPUT 记录的含义：每一行都能回到一篇文章、一个句子和一个实体；**原始文本、字符位置和标准编号共同构成可复核的证据。**
> 从文本处理到知识发现：NLTK 并没有直接得出"STK11 导致某疾病"的结论；它只是帮助我们建立稳定、可检查的证据单元。
> 课堂思考：如果一个实体跨越两个句子，或者句子切分发生错误，应该如何标记、复核和修正？

**案例四：用 NLTK 筛选 RiceMind 句子证据**

对比：PubTator 需要字符偏移对齐；RiceMind 已给出句子级证据（sent_id、text），NLTK 的任务从"对齐"变为"筛选与统计"。

```python
import json
from pathlib import Path
from nltk.tokenize import word_tokenize

hits = []
for path in sorted(Path(".").glob("sd1.evidence.page*.json")):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in data["results"]:      # 键名以 API 实际返回为准
        text = item["text"]
        words = [w.lower() for w in word_tokenize(text)]
        if "drought" in words:
            hits.append({"PMID": item["PMID"], "sent_id": item["sent_id"], "text": text})

for pmid, sent_id, text in hits:
    print(pmid, sent_id, text, sep="\t")
```

> **与 PubTator 流程的分工差异**：PubTator 提供原始标注，NLTK 负责建立句子证据单元；RiceMind 提供已结构化的句子证据，NLTK 负责词级筛选与统计。两者最终都输出"**PMID + sent_id + text**"的可复核记录。

### 4.4 验证与课堂练习（p54–56）

| CHECK                        | LIMITATION                      |
| ---------------------------- | ------------------------------- |
| 句子数量是否合理？           | 缩写和小数可能造成错误切分      |
| 每个实体是否仍在原句中？     | 基因符号可能被当作普通词        |
| begin/end 是否对应正确字符？ | 连字符和括号会影响词元化        |
| PMID 和实体编号是否保留？    | 通用 NLTK 模型不等于专业 BioNER |
| 抽样阅读结果是否可信？       |                                 |

> 本章结论：可复现的数据分析不仅需要代码，还需要明确的输入、结构化的输出和**可解释的验证步骤**。
> （核心观念：**代码运行不等于结果正确**。）

**课堂练习：从关键词到证据表**

- 给定：一批 PubTator 摘要、一个关键词集合和实体类型集合。
- 请完成：
  1. 使用 NLTK 切分句子；
  2. 为每个句子分配 `sent_id`；
  3. 找出包含目标关键词的句子；
  4. 输出 PMID、句子编号、句子文本、实体提及、实体类型和标准编号；
  5. 随机抽取若干结果，检查句子边界和实体对齐。
- 课堂思考：提交结果时，不只提交代码；还要提交**一条输入样例、一条输出样例和一条你发现的错误或局限**。

---

## 5. 讨论：面向水稻数据的自主实体识别开发（p57–63）

### 领域缺口

| PubTator 覆盖的实体类型                               | RiceMind 需要的实体                                                                                |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| gene、disease、chemical、species、mutation、cell line | 作物性状条目（grain yield、drought tolerance、tillering……，对应 TO/CO 本体）、品种名、QTL 位点等 |
| ——为人类医学文献训练的模型                          | ——**PubTator 完全没有这一类**                                                              |

> 回到开头的工具地图：当 API 的领域知识**不够用**时，缺口要由自己的代码补上：**从 API 的使用者变成工具的构建者。**

### 三条路线

| 路线          | 做法                                                         | 优点                       | 代价                   |
| ------------- | ------------------------------------------------------------ | -------------------------- | ---------------------- |
| A. 词典匹配   | RiceMind 拉性状表建词典；NLTK 分词 + n-gram 匹配；对齐到句子 | 零训练数据、可解释、可复现 | 同义词覆盖不全         |
| B. 共现拼接   | PubTator 抽基因，词典抽性状，句子级共现                      | 各取所长，改动最小         | 共现 ≠ 关联，需过滤   |
| C. 弱监督扩展 | 用 RiceMind 句子作种子，提取上下文模式，迭代扩词表           | 能发现词典外的新性状词     | 工程量超出 NLTK 舒适区 |

> 路线 A 与 B 的零件全部是本课已讲过的：API 下载、NLTK 分词、句子对齐、证据表。路线 C 作为讨论方向，点到为止。

### 示例：词典匹配路线

- 输入：RiceMind 性状词表 + PubMed 摘要文本。
- 示例逻辑：
  - 从 RiceMind 取性状词表：`["grain yield", "drought tolerance", ...]`；
  - NLTK 对摘要做切句与分词，生成 1–3 元 n-gram；
  - n-gram 与词典（小写化后）匹配，命中即得性状提及；
  - 复用 `sentence_spans` 把提及对齐到 `sent_id`，输出证据行。
- 示例输出：`PMID + sent_id + "grain yield" + Trait + TO:0000397`

### 示例：共现拼接路线

- 输入：同一批 PubMed 摘要，PubTator API 在线标注。
- 示例逻辑：
  - PubTator 照常提取基因提及（自带 GeneID）；
  - 路线 A 的词典抽取同一文本中的性状提及；
  - 两者都通过句子 span 对齐到 `sent_id`；
  - 同一句内基因与性状共现 → 候选"基因-性状"对。
- **边界**：共现不等于关联——"OsX increases grain yield" 与 "unlike OsX, grain yield ……" 在共现层面无法区分，需要规则或人工过滤。

### 评估设计

- RiceMind 的策展证据（PMID + 句子 + 基因 + 性状）作为**金标准**；
- 自建 pipeline 的输出与之对齐，计算 precision / recall；
- 错误分析：漏同义词 → recall 低；"yield" 歧义 → precision 低。

> 开放讨论：词典法的天花板在哪里？换成玉米/小麦数据，哪些代码可以直接复用？什么情况下应该放弃 NLTK 规则、转向训练模型？

### 本节结论

**PubTator、Shell、RiceMind、NLTK 四条线在此汇合**：API 提供标注与金标准，NLTK 补上领域缺口，评估驱动迭代。

---

## 全讲一句话总结

用 Shell 从 PubMed/PubTator 批量拉取带实体标注的文献，用 RiceMind API 获取水稻领域证据，用 NLTK 把带字符偏移的标注加工成"可复核的句子级证据表"，并进一步讨论了如何自建面向水稻的实体识别 pipeline（词典匹配 → 共现拼接 → 弱监督扩展）。
