"""
约束解析器：将自然语言场景文本转化为结构化约束

实现两种解析策略：
1. LLM-based: 使用微调后的模型进行解析（主方案）
2. Template-based: 基于规则模板的解析（回退方案）
"""
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from src.constraint_schema import (
    StructuredConstraints, Domain,
    TimeConstraints, AbsoluteTimeConstraint, RelativeTimeConstraint,
    SpaceConstraints, SpatialStructure, PositionConstraint, SpatialRelationConstraint,
    SocialConstraints, KinshipConstraint,
    NatureConstraints, PropertyConstraint, CategoryConstraint,
)


# ============================================================
# LLM Prompt 模板
# ============================================================

SYSTEM_PROMPT = """You are an expert at analyzing commonsense reasoning problems and extracting structured constraints from natural language descriptions.

Your task is to read a scenario description and extract all logical constraints into a structured JSON format.
The constraints must be precise enough for a constraint solver to solve the problem deterministically.

Output ONLY valid JSON, no explanation."""


TIME_PROMPT = """## Task: Temporal Constraint Extraction

Given a temporal reasoning scenario, extract all events and their temporal relationships.

### Output Schema:
```json
{
  "domain": "time",
  "entities": ["event1", "event2", ...],
  "time": {
    "entities": ["event1", "event2", ...],
    "is_weekly_cycle": true/false,
    "absolute": [
      {"event": "event_name", "time_point": "周三"}
    ],
    "relative": [
      {"event_a": "event_name", "event_b": "event_name", "relation": "after|before", "offset": 1, "unit": "day"}
    ]
  }
}
```

### Rules:
- "event_a N days after event_b" → relation="after", offset=N
- "event_a N days before event_b" → relation="before", offset=N
- Weekday names are absolute time points
- For weekly scenarios, set is_weekly_cycle=true
- Extract ALL mentioned events, including those in options

### Scenario:
{text}

### Question (for context):
{question}

### Output the structured constraints:"""


SPACE_PROMPT = """## Task: Spatial Constraint Extraction

Given a spatial reasoning scenario, extract all entities and their spatial relationships.

### Output Schema:
```json
{
  "domain": "space",
  "entities": ["entity1", "entity2", ...],
  "space": {
    "structure": "grid_3x2",
    "rows": 3,
    "cols": 2,
    "col_labels": ["东", "西"],
    "entities": ["entity1", "entity2", ...],
    "positions": [
      {"entity": "entity_name", "row": 1, "col": "东"}
    ],
    "relations": [
      {"entity_a": "entity1", "entity_b": "entity2", "relation": "above|below|adjacent_left|adjacent_right|same_row|same_col|different_row|different_col", "gap": 0}
    ]
  }
}
```

### Rules:
- "A在B正上方且隔了N层" → relation="above", gap=N
- "A是B的左邻/右邻" → relation="adjacent_left"/"adjacent_right"
- "A在B右上方" → A is up-right of B (combine with position deduction)
- "N层东侧/西侧是X" → position with row=N, col=东/西
- 东=左, 西=右

### Scenario:
{text}

### Output the structured constraints:"""


SOCIAL_PROMPT = """## Task: Social Relationship Constraint Extraction

Given a social reasoning scenario, extract all people and their relationships.

### Output Schema:
```json
{
  "domain": "social",
  "entities": ["person1", "person2", ...],
  "social": {
    "entities": ["person1", "person2", ...],
    "relations": [
      {"person_a": "person1", "person_b": "person2", "relation": "父亲"}
    ]
  }
}
```

### Rules:
- "A是B的X" → person_a=A, person_b=B, relation=X
- "A is B's X" → same format
- Extract ALL named persons in the text
- Both Chinese and English kinship/social terms

### Scenario:
{text}

### Output the structured constraints:"""


NATURE_PROMPT = """## Task: Natural Commonsense Constraint Extraction

Given a natural attribute reasoning scenario, extract all entities, their properties, and constraints.

### Output Schema:
```json
{
  "domain": "nature",
  "entities": ["entity1", "entity2", ...],
  "nature": {
    "entities": ["entity1", "entity2", ...],
    "positions": ["position1", "position2", ...],
    "entity_properties": {
      "entity1": {"property1": "value1", "property2": "value2"}
    },
    "property_constraints": [
      {"position": "position1", "property_name": "花色", "property_value": "黄色"}
    ],
    "category_constraints": [
      {"position": "position1", "category": "蔬菜"}
    ]
  }
}
```

### Rules:
- "N号田的作物属于X类" → category_constraint with category=X
- "N号田的作物开的花是Y色" → property_constraint
- "N号田的作物可食用部分是X" → property_constraint
- Include known commonsense properties of entities in entity_properties

### Scenario:
{text}

### Output the structured constraints:"""


# ============================================================
# 辅助函数
# ============================================================

def _split_numbered_items(text: str) -> List[str]:
    """将文本按编号 (1) (2) 等分割为独立的约束项"""
    # 先按换行分割
    lines = text.split('\n')
    items = []
    preamble = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'\(\d+\)', line):
            # 提取编号后的内容
            content = re.sub(r'\(\d+\)\s*', '', line).strip()
            items.append(content)
        else:
            preamble.append(line)

    # 如果文本中没有 (N) 编号，尝试用分号分割
    if not items and text:
        # 用于中文社交等没有编号的文本
        pass

    return items, ' '.join(preamble)


def _extract_event_name(text: str, is_en: bool = False) -> str:
    """清理事件名称"""
    # 去除末尾标点和空格
    text = text.strip()
    if is_en:
        text = re.sub(r'[;,.]+$', '', text)
    else:
        text = re.sub(r'[；。，、;,.]+$', '', text)
    return text.strip()


# ============================================================
# 解析器类
# ============================================================

class ConstraintParser:
    """约束解析器基类"""

    def parse(self, text: str, domain: str, question: str = "",
              language: str = "cn") -> StructuredConstraints:
        raise NotImplementedError


class LLMConstraintParser(ConstraintParser):
    """基于LLM的约束解析器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def parse(self, text: str, domain: str, question: str = "",
              language: str = "cn") -> StructuredConstraints:
        """使用LLM解析约束"""
        prompt = self._build_prompt(text, domain, question)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取JSON部分
        constraints = self._extract_json(response)
        return self._dict_to_constraints(constraints, domain, text, language)

    def _build_prompt(self, text: str, domain: str, question: str) -> str:
        """构建领域特定的prompt"""
        base_domain = domain.split("+")[0]

        domain_prompts = {
            "time": TIME_PROMPT,
            "space": SPACE_PROMPT,
            "social": SOCIAL_PROMPT,
            "nature": NATURE_PROMPT,
        }

        template = domain_prompts.get(base_domain, TIME_PROMPT)
        return template.format(text=text, question=question)

    def _extract_json(self, response: str) -> Dict:
        """从LLM响应中提取JSON"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        brace_match = re.search(r'\{.*\}', response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def _dict_to_constraints(self, data: Dict, domain: str,
                              text: str, language: str) -> StructuredConstraints:
        """将字典转化为StructuredConstraints对象"""
        sc = StructuredConstraints(domain=domain, language=language, raw_text=text)

        if "time" in data:
            tc_data = data["time"]
            sc.time = TimeConstraints(
                entities=tc_data.get("entities", []),
                is_weekly_cycle=tc_data.get("is_weekly_cycle", False),
                absolute=[AbsoluteTimeConstraint(**a) for a in tc_data.get("absolute", [])],
                relative=[RelativeTimeConstraint(**r) for r in tc_data.get("relative", [])],
            )

        if "space" in data:
            sp_data = data["space"]
            sc.space = SpaceConstraints(
                structure=SpatialStructure(sp_data.get("structure", "grid_3x2")),
                rows=sp_data.get("rows", 3),
                cols=sp_data.get("cols", 2),
                col_labels=sp_data.get("col_labels", ["东", "西"]),
                entities=sp_data.get("entities", []),
                positions=[PositionConstraint(**p) for p in sp_data.get("positions", [])],
                relations=[SpatialRelationConstraint(**r) for r in sp_data.get("relations", [])],
            )

        if "social" in data:
            soc_data = data["social"]
            sc.social = SocialConstraints(
                entities=soc_data.get("entities", []),
                relations=[KinshipConstraint(**r) for r in soc_data.get("relations", [])],
            )

        if "nature" in data:
            nat_data = data["nature"]
            sc.nature = NatureConstraints(
                entities=nat_data.get("entities", []),
                positions=nat_data.get("positions", []),
                entity_properties=nat_data.get("entity_properties", {}),
                property_constraints=[PropertyConstraint(**p) for p in nat_data.get("property_constraints", [])],
                category_constraints=[CategoryConstraint(**c) for c in nat_data.get("category_constraints", [])],
            )

        return sc


class TemplateConstraintParser(ConstraintParser):
    """
    基于模板的约束解析器（回退方案）
    使用正则表达式提取约束，不需要LLM
    """

    def parse(self, text: str, domain: str, question: str = "",
              language: str = "cn") -> StructuredConstraints:
        """使用模板解析约束"""
        base_domain = domain.split("+")[0]

        if base_domain == "time":
            return self._parse_time(text, question, language)
        elif base_domain == "space":
            return self._parse_space(text, question, language)
        elif base_domain == "social":
            return self._parse_social(text, question, language)
        elif base_domain == "nature":
            return self._parse_nature(text, question, language)
        else:
            return StructuredConstraints(domain=domain, language=language, raw_text=text)

    # ================================================================
    # 时间域解析
    # ================================================================

    def _parse_time(self, text: str, question: str,
                    language: str) -> StructuredConstraints:
        """模板解析时间约束"""
        sc = StructuredConstraints(domain="time", language=language, raw_text=text)
        tc = TimeConstraints(is_weekly_cycle=False)
        entities = set()
        absolute_times = []
        relative_times = []

        is_en = (language == "en")

        if is_en:
            self._parse_time_en(text, entities, absolute_times, relative_times, tc)
        else:
            self._parse_time_cn(text, entities, absolute_times, relative_times, tc)

        tc.entities = list(entities)
        tc.absolute = absolute_times
        tc.relative = relative_times
        sc.time = tc
        return sc

    def _parse_time_en(self, text: str, entities: set, absolute_times: list,
                       relative_times: list, tc: TimeConstraints):
        """解析英文时间约束"""
        # 检测是否每周循环
        if re.search(r'every\s+week|weekly|broadcasted\s+at\s+fixed\s+times\s+every\s+week', text, re.IGNORECASE):
            tc.is_weekly_cycle = True

        # 提取主体名称（用于代词消解）
        subject_name = "Jack"
        subj_match = re.search(r"(\w+(?:\s+\w+)?)'s\s+(?:daughter|son|wife|husband|friend|life)", text)
        if subj_match:
            subject_name = subj_match.group(1)

        # 按换行分割（每个 (N) 项占一行）
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # 去掉编号前缀
            line_clean = re.sub(r'^\(\d+\)\s*', '', line).strip()

            # 清理末尾分号
            line_clean = re.sub(r'[;；]+$', '', line_clean).strip()

            # 代词消解: 将 "he/his/him" 替换为主题名
            line_clean = re.sub(r'\bhe\s+(?=met|started|became|retired|graduated|studied|was|lived|got|passed)',
                               f'{subject_name} ', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bhe\b', subject_name, line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bhis\b', f"{subject_name}'s", line_clean, flags=re.IGNORECASE)

            # 模式: "X N years before Y"
            m = re.match(r'^(.+?)\s+(\d+)\s+years?\s+(before|after)\s+(.+?)$', line_clean)
            if m:
                event_x = _extract_event_name(m.group(1), is_en=True)
                offset = int(m.group(2))
                relation = m.group(3)
                event_y = _extract_event_name(m.group(4), is_en=True)

                if relation == "after":
                    relative_times.append(RelativeTimeConstraint(
                        event_a=event_x, event_b=event_y,
                        relation="after", offset=offset, unit="year"
                    ))
                else:
                    relative_times.append(RelativeTimeConstraint(
                        event_a=event_x, event_b=event_y,
                        relation="before", offset=offset, unit="year"
                    ))
                entities.add(event_x)
                entities.add(event_y)
                continue

            # 模式: "X N days after Y" (融合域)
            m = re.match(r'^(.+?)\s+(\d+)\s*days?\s+(after|before)\s+(.+?)$', line_clean)
            if m:
                event_x = _extract_event_name(m.group(1), is_en=True)
                offset = int(m.group(2))
                relation = m.group(3)
                event_y = _extract_event_name(m.group(4), is_en=True)

                if relation == "after":
                    relative_times.append(RelativeTimeConstraint(
                        event_a=event_x, event_b=event_y,
                        relation="after", offset=offset, unit="day"
                    ))
                else:
                    relative_times.append(RelativeTimeConstraint(
                        event_a=event_x, event_b=event_y,
                        relation="before", offset=offset, unit="day"
                    ))
                entities.add(event_x)
                entities.add(event_y)
                continue

            # 模式: "In YYYY, X..."
            m = re.match(r'^[Ii]n\s+(\d{4})\s*[,，]?\s*(.+?)$', line_clean)
            if m:
                year = m.group(1)
                event = _extract_event_name(m.group(2), is_en=True)
                absolute_times.append(AbsoluteTimeConstraint(event=event, time_point=year))
                entities.add(event)
                continue

    def _parse_time_cn(self, text: str, entities: set, absolute_times: list,
                       relative_times: list, tc: TimeConstraints):
        """解析中文时间约束"""
        # 检测每周循环
        if re.search(r'每周|星期|周[一二三四五六日]', text):
            tc.is_weekly_cycle = True

        # 按项目分割: 通过 (N) 或换行
        items = re.split(r'\n', text)

        for item in items:
            item = item.strip()
            if not item:
                continue
            # 去掉编号
            item = re.sub(r'^\(\d+\)\s*', '', item).strip()

            # 模式1: 绝对时间 "(1)周三，他打羽毛球;" 或 "(1)星期三，他开组会;"
            m = re.match(r'(周[一二三四五六日天]|星期[一二三四五六日天])\s*[,，]?\s*(.+?)[;；。]*$', item)
            if m:
                time_point = m.group(1)
                event = _extract_event_name(m.group(2))
                absolute_times.append(AbsoluteTimeConstraint(event=event, time_point=time_point))
                entities.add(event)
                continue

            # 模式2: 相对时间 "在他开组会之后1天，他阅读科幻小说"
            m = re.match(r'在\s*(.+?)\s*[之以]?\s*(前|后)\s*(\d+)\s*天\s*[,，]?\s*(.+?)[;；。]*$', item)
            if m:
                ref_event = m.group(1).strip()  # 参考事件
                direction = m.group(2)  # 前/后
                offset = int(m.group(3))
                target_event = _extract_event_name(m.group(4))

                if direction == "后":
                    # 在X之后N天，Y → Y after X
                    relative_times.append(RelativeTimeConstraint(
                        event_a=target_event, event_b=ref_event,
                        relation="after", offset=offset, unit="day"
                    ))
                else:
                    # 在X之前N天，Y → Y before X
                    relative_times.append(RelativeTimeConstraint(
                        event_a=target_event, event_b=ref_event,
                        relation="before", offset=offset, unit="day"
                    ))
                entities.add(ref_event)
                entities.add(target_event)
                continue

            # 模式3: X之后N天，Y (无"在"前缀)
            m = re.match(r'(.+?)\s*之后\s*(\d+)\s*天\s*[,，]?\s*(.+?)[;；。]*$', item)
            if m:
                ref_event = m.group(1).strip()
                offset = int(m.group(2))
                target_event = _extract_event_name(m.group(3))
                relative_times.append(RelativeTimeConstraint(
                    event_a=target_event, event_b=ref_event,
                    relation="after", offset=offset, unit="day"
                ))
                entities.add(ref_event)
                entities.add(target_event)
                continue

            # 模式4: X之前N天，Y
            m = re.match(r'(.+?)\s*之前\s*(\d+)\s*天\s*[,，]?\s*(.+?)[;；。]*$', item)
            if m:
                ref_event = m.group(1).strip()
                offset = int(m.group(2))
                target_event = _extract_event_name(m.group(3))
                relative_times.append(RelativeTimeConstraint(
                    event_a=target_event, event_b=ref_event,
                    relation="before", offset=offset, unit="day"
                ))
                entities.add(ref_event)
                entities.add(target_event)
                continue

            # 模式5: X的N天之后，Y / 在X的N天之后，Y
            m = re.match(r'(?:在\s*)?(.+?)\s*的\s*(\d+)\s*天\s*之后\s*[,，]?\s*(.+?)[;；。]*$', item)
            if m:
                ref_event = m.group(1).strip()
                offset = int(m.group(2))
                target_event = _extract_event_name(m.group(3))
                relative_times.append(RelativeTimeConstraint(
                    event_a=target_event, event_b=ref_event,
                    relation="after", offset=offset, unit="day"
                ))
                entities.add(ref_event)
                entities.add(target_event)
                continue

            # 模式5b: X的N天之前，Y
            m = re.match(r'(?:在\s*)?(.+?)\s*的\s*(\d+)\s*天\s*之前\s*[,，]?\s*(.+?)[;；。]*$', item)
            if m:
                ref_event = m.group(1).strip()
                offset = int(m.group(2))
                target_event = _extract_event_name(m.group(3))
                relative_times.append(RelativeTimeConstraint(
                    event_a=target_event, event_b=ref_event,
                    relation="after", offset=offset, unit="day"
                ))
                entities.add(ref_event)
                entities.add(target_event)
                continue

    # ================================================================
    # 空间域解析
    # ================================================================

    def _parse_space(self, text: str, question: str,
                     language: str) -> StructuredConstraints:
        """模板解析空间约束"""
        sc = StructuredConstraints(domain="space", language=language, raw_text=text)
        sp_c = SpaceConstraints(structure=SpatialStructure.GRID_3X2, rows=3, cols=2)

        # 提取实体（常见花卉/商品名称）
        entity_patterns = [
            r'(茶花|水仙|波斯菊|月季|君子兰|郁金香|牡丹|芍药|菊花|兰花|玫瑰|百合|杜鹃|梅花)',
            r'(棉花糖|创可贴|大麦|牛乳|三明治|担担面|剪刀|锤子|面包|牛奶|果汁)',
            r'(clivia miniata|hot-water bottle|catfish|toast|epiphyllum|barley|haw|pomegranate|rose|lily|tulip)',
        ]
        entities = set()
        for pattern in entity_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.add(m.group(1))
        sp_c.entities = list(entities)

        # 按约束项分析 - 通常以"已知："为界
        known_part = text
        if "已知：" in text:
            known_part = text.split("已知：")[1] if len(text.split("已知：")) > 1 else text
        elif "已知:" in text:
            known_part = text.split("已知:")[1] if len(text.split("已知:")) > 1 else text

        # 分隔约束项 (中文逗号和分号都可能是分隔符)
        items = re.split(r'[；;。，,]\s*', known_part)

        for item in items:
            item = item.strip()
            if not item or len(item) < 4:
                continue

            # 模式1: "A在B正上方且(二者)?隔了N层"
            m = re.search(r'(\S+?)在(\S+?)正上方且(?:二者)?隔了([一二三四五六七八九十\d]+)层', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="above", gap=self._parse_chinese_num(m.group(3))
                ))
                continue

            # 模式2: "A是B的左邻" / "A是B的右邻"
            m = re.search(r'(\S+?)是(\S+?)的(左|右)邻', item)
            if m:
                relation = "adjacent_left" if m.group(3) == "左" else "adjacent_right"
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2), relation=relation
                ))
                continue

            # 模式3: "A在B右上方且(二者)?隔了N层"
            m = re.search(r'(\S+?)在(\S+?)右上方且(?:二者)?隔了([一二三四五六七八九十\d]+)层', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="above", gap=self._parse_chinese_num(m.group(3))
                ))
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="different_col", gap=0
                ))
                continue

            # 模式4: "A在B下方且隔了N层"
            m = re.search(r'(\S+?)在(\S+?)下方且(?:二者)?隔了([一二三四五六七八九十\d]+)层', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="below", gap=self._parse_chinese_num(m.group(3))
                ))
                continue

            # 模式5: "A在B左上方且隔了N层"
            m = re.search(r'(\S+?)在(\S+?)左上方且(?:二者)?隔了([一二三四五六七八九十\d]+)层', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="above", gap=self._parse_chinese_num(m.group(3))
                ))
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="different_col", gap=0
                ))
                continue

            # 模式6: "A在N层" / "A在顶层/底层/中间层"
            m = re.search(r'(\S+?)在(?:第?(\d+)|顶|底|中间)层', item)
            if m:
                entity = m.group(1)
                if m.group(2) and m.group(2).isdigit():
                    row = int(m.group(2)) - 1  # 1-based to 0-based
                elif "顶" in m.group(0):
                    row = 0
                elif "底" in m.group(0):
                    row = 2
                elif "中间" in m.group(0):
                    row = 1
                else:
                    row = 1
                sp_c.positions.append(PositionConstraint(entity=entity, row=row))
                continue

            # 模式7: "中间层东侧是X" / "第N层东侧/西侧是X" / "顶层东侧是X"
            m = re.search(r'(?:第(\d+)|中间|顶|底)层(东|西)侧是(\S+)', item)
            if m:
                entity = m.group(3)
                if m.group(1) and m.group(1).isdigit():
                    row = int(m.group(1)) - 1
                elif "顶" in m.group(0):
                    row = 0
                elif "底" in m.group(0):
                    row = 2
                else:
                    row = 1
                col = m.group(2)
                sp_c.positions.append(PositionConstraint(entity=entity, row=row, col=col))
                continue

            # 模式8: "A所在层和B所在层相邻"
            m = re.search(r'(\S+?)所在层和(\S+?)所在层相邻', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="above_or_below", gap=0
                ))
                continue

            # 模式9: "A在B上一层"
            m = re.search(r'(\S+?)在(\S+?)上(?:一)?层', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="above", gap=0
                ))
                continue

            # 模式10: "A和B不在同一层" / "A与B不同层"
            m = re.search(r'(\S+?)(?:和|与)(\S+?)(?:不在同一层|不同层)', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="different_row", gap=0
                ))
                continue

            # 模式11: "A和B不在同一侧" / "A与B不同侧"
            m = re.search(r'(\S+?)(?:和|与)(\S+?)(?:不在同一侧|不同侧)', item)
            if m:
                sp_c.relations.append(SpatialRelationConstraint(
                    entity_a=m.group(1), entity_b=m.group(2),
                    relation="different_col", gap=0
                ))
                continue

        sc.space = sp_c
        return sc

    # ================================================================
    # 社会域解析
    # ================================================================

    def _parse_social(self, text: str, question: str,
                       language: str) -> StructuredConstraints:
        """模板解析社会关系约束"""
        sc = StructuredConstraints(domain="social", language=language, raw_text=text)
        soc_c = SocialConstraints()
        entities = set()
        relations = []

        if language == "cn":
            # 提取中文姓名：2-3个汉字
            # 使用更灵活的姓名模式
            name_chars = set()
            # 从"已知："后提取所有关系
            known_text = text
            if "已知：" in text:
                known_text = text.split("已知：")[1] if len(text.split("已知：")) > 1 else text

            # Split items by commas/semicolons
            items = re.split(r'[；;，,。]\s*', known_text)

            for item in items:
                item = item.strip()
                if not item:
                    continue

                # "A是B的C" 格式
                m = re.match(r'(\S{2,4})是(\S{2,4})的(.+)', item)
                if m:
                    a, b, rel = m.group(1), m.group(2), m.group(3).strip()
                    # 清理关系名称 (去掉后续的"也是"等)
                    rel = re.sub(r'[,，]\s*也是.*$', '', rel)
                    rel = re.sub(r'[；;。]$', '', rel)
                    entities.add(a)
                    entities.add(b)
                    relations.append(KinshipConstraint(person_a=a, person_b=b, relation=rel))
                    continue

                # "A也是B的C" 格式
                m = re.match(r'(\S{2,4})也是(\S{2,4})的(.+)', item)
                if m:
                    a, b, rel = m.group(1), m.group(2), m.group(3).strip()
                    rel = re.sub(r'[,，]\s*也是.*$', '', rel)
                    rel = re.sub(r'[；;。]$', '', rel)
                    entities.add(a)
                    entities.add(b)
                    relations.append(KinshipConstraint(person_a=a, person_b=b, relation=rel))
        else:
            # "A is B's C" 格式
            for m in re.finditer(r"(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)'s\s+([\w\s-]+?)(?=[,;.]|$|\s+(?:and|also|,))", text):
                a, b, rel = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                entities.add(a)
                entities.add(b)
                relations.append(KinshipConstraint(person_a=a, person_b=b, relation=rel))
            # "A, B's C, ..." 格式
            for m in re.finditer(r"(\w+(?:\s+\w+)?),\s+(\w+(?:\s+\w+)?)'s\s+([\w\s-]+?)(?=[,;.]|$|\s+(?:and|also|,))", text):
                a, b, rel = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                entities.add(a)
                entities.add(b)
                relations.append(KinshipConstraint(person_a=a, person_b=b, relation=rel))

        soc_c.entities = list(entities)
        soc_c.relations = relations
        sc.social = soc_c
        return sc

    # ================================================================
    # 自然域解析
    # ================================================================

    def _parse_nature(self, text: str, question: str,
                       language: str) -> StructuredConstraints:
        """模板解析自然常识约束"""
        sc = StructuredConstraints(domain="nature", language=language, raw_text=text)
        nat_c = NatureConstraints()
        entities = set()
        positions = set()

        # 提取位置（N号田/N号格/photo No.N）
        for m in re.finditer(r'(\d+号[田格位])', text):
            positions.add(m.group(1))
        for m in re.finditer(r'photo\s+No\.(\d+)', text, re.IGNORECASE):
            positions.add(f"photo No.{m.group(1)}")

        # 提取实体名称
        chinese_entities_pattern = r'(莴苣|南瓜|胡萝卜|花生|土豆|白菜|菠菜|西红柿|苹果|梨|茶花|水仙|波斯菊|月季|君子兰|郁金香|牡丹|芍药|菊花|兰花|大麦|小麦|水稻|玉米|棉花糖|三明治|剪刀|锤子|牛乳|创可贴)'
        for m in re.finditer(chinese_entities_pattern, text):
            entities.add(m.group(1))

        english_entities_pattern = r'\b(epiphyllum|barley|haw|pomegranate|catfish|goldfish|salmon|tuna|clivia miniata|hot-water bottle|toast|rose|lily|tulip)\b'
        for m in re.finditer(english_entities_pattern, text, re.IGNORECASE):
            entities.add(m.group(1).lower())

        nat_c.entities = list(entities)
        nat_c.positions = list(positions)

        # 按分号分隔约束项
        known_text = text
        if "已知：" in text:
            known_text = text.split("已知：")[1] if len(text.split("已知：")) > 1 else text
        elif "已知:" in text:
            known_text = text.split("已知:")[1] if len(text.split("已知:")) > 1 else text
        elif "It is known that:" in text:
            known_text = text.split("It is known that:")[1] if len(text.split("It is known that:")) > 1 else text

        items = re.split(r'[；;。]\s*', known_text)

        for item in items:
            item = item.strip()
            if not item or len(item) < 4:
                continue

            # 去掉编号
            item = re.sub(r'^\(\d+\)\s*', '', item).strip()

            # 模式1: "N号田中的作物K是V" / "N号田中作物的K是V"
            m = re.match(r'(\d+号[田格位])(?:中|的|中的)?(?:作物)?(?:的)?(\S+?)是(\S+)', item)
            if m:
                pos = m.group(1)
                prop_name = m.group(2)
                prop_value = m.group(3).rstrip('。，;')
                # 判断是类别约束还是属性约束
                if prop_name in ('属于',):
                    nat_c.category_constraints.append(CategoryConstraint(
                        position=pos, category=prop_value
                    ))
                else:
                    nat_c.property_constraints.append(PropertyConstraint(
                        position=pos, property_name=prop_name, property_value=prop_value
                    ))
                continue

            # 模式2: "N号X属于C类"
            m = re.match(r'(\d+号[田格位])(?:的?\S*?)?属于(\S+)', item)
            if m:
                pos = m.group(1)
                cat = m.group(2).rstrip('。，;')
                nat_c.category_constraints.append(CategoryConstraint(position=pos, category=cat))
                continue

            # 模式3: "N号X与M号Y的Z相同" → equality constraint
            m = re.match(r'(\d+号[田格位])(?:中的?(?:作物)?)?(?:与|和)(\d+号[田格位])(?:中的?(?:作物)?)?的?(\S+?)相同', item)
            if m:
                pos1 = m.group(1)
                pos2 = m.group(2)
                prop = m.group(3)
                # 标准化属性名
                normalized = self._norm_prop_name(prop)
                # 添加一个约束：pos1和pos2在此属性上取值相同
                # 用category_constraint来传递此信息（position=pos1, category="same_<prop>_as_<pos2>"）
                nat_c.category_constraints.append(CategoryConstraint(
                    position=pos1, category=f"same_{normalized}_as_{pos2}"
                ))
                continue

            # 模式4: "N号X与M号Y的作物开的花颜色相同" → more specific equality
            m = re.match(r'(\d+号[田格位])(?:中的?(?:作物)?)?(?:与|和)(\d+号[田格位])(?:中的?(?:作物)?)?(?:的)?(?:作物)?开?的?(?:花)?\s*(\S+?)相同', item)
            if m:
                pos1 = m.group(1)
                pos2 = m.group(2)
                prop = m.group(3)
                normalized = self._norm_prop_name(prop)
                nat_c.category_constraints.append(CategoryConstraint(
                    position=pos1, category=f"same_{normalized}_as_{pos2}"
                ))
                continue

        # 添加常识属性
        self._add_commonsense_properties(nat_c)

        sc.nature = nat_c
        return sc

    @staticmethod
    def _parse_chinese_num(text: str) -> int:
        """将中文数字转为整数"""
        cn_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                   '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                   '零': 0, '两': 2}
        if text in cn_nums:
            return cn_nums[text]
        try:
            return int(text)
        except ValueError:
            return 0

    @staticmethod
    def _norm_prop_name(name: str) -> str:
        """标准化属性名称"""
        name = name.strip()
        mappings = {
            "颜色": "花色", "花色": "花色", "花": "花色",
            "color": "flower_color", "flower color": "flower_color",
        }
        return mappings.get(name, name)

    def _add_commonsense_properties(self, nat_c: NatureConstraints):
        """添加已知的常识属性"""
        COMMONSENSE = {
            "莴苣": {"类别": "蔬菜", "花色": "?", "可食用部分": "茎叶"},
            "南瓜": {"类别": "蔬菜", "花色": "黄色", "可食用部分": "果实"},
            "胡萝卜": {"类别": "蔬菜", "花色": "白色", "可食用部分": "根"},
            "花生": {"类别": "蔬菜", "花色": "黄色", "可食用部分": "种子"},
            "土豆": {"类别": "蔬菜", "花色": "白色", "可食用部分": "块茎"},
            "白菜": {"类别": "蔬菜", "花色": "黄色", "可食用部分": "叶片"},
            "菠菜": {"类别": "蔬菜", "花色": "?", "可食用部分": "叶片"},
            "西红柿": {"类别": "蔬菜", "花色": "黄色", "可食用部分": "果实"},
            "苹果": {"类别": "水果", "花色": "白色", "可食用部分": "果实"},
            "梨": {"类别": "水果", "花色": "白色", "可食用部分": "果实"},
            "茶花": {"类别": "花卉", "花色": "红色", "可食用部分": "?"},
            "水仙": {"类别": "花卉", "花色": "白色", "可食用部分": "?"},
            "波斯菊": {"类别": "花卉", "花色": "多种", "可食用部分": "?"},
            "月季": {"类别": "花卉", "花色": "红色", "可食用部分": "?"},
            "君子兰": {"类别": "花卉", "花色": "橙色", "可食用部分": "?"},
            "郁金香": {"类别": "花卉", "花色": "多种", "可食用部分": "?"},
            "牡丹": {"类别": "花卉", "花色": "红色", "可食用部分": "?"},
            "芍药": {"类别": "花卉", "花色": "粉红", "可食用部分": "?"},
            "菊花": {"类别": "花卉", "花色": "黄色", "可食用部分": "?"},
            "兰花": {"类别": "花卉", "花色": "多种", "可食用部分": "?"},
            "大麦": {"类别": "谷物", "花色": "?", "可食用部分": "种子"},
            "小麦": {"类别": "谷物", "花色": "?", "可食用部分": "种子"},
            "水稻": {"类别": "谷物", "花色": "?", "可食用部分": "种子"},
            "玉米": {"类别": "谷物", "花色": "?", "可食用部分": "种子"},
            "epiphyllum": {"category": "flower", "flower_color": "white"},
            "barley": {"category": "grain", "flower_color": "?"},
            "haw": {"category": "fruit", "flower_color": "white", "taste": "sour", "color": "red"},
            "pomegranate": {"category": "fruit", "flower_color": "red", "taste": "sweet", "color": "red"},
            "catfish": {"category": "fish"},
            "toast": {"category": "processed food"},
            "clivia miniata": {"category": "flower"},
        }
        for entity in nat_c.entities:
            if entity in COMMONSENSE:
                nat_c.entity_properties[entity] = COMMONSENSE[entity]
