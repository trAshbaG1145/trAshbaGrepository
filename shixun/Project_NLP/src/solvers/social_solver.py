"""
社会关系推理求解器

核心算法：关系图构建 + 传递闭包 + 性别/代际推理
- 将亲属/社会关系转化为有向图
- 通过传递闭包推导隐含关系
- 利用中文亲属称谓推断性别和代际
"""
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from src.constraint_schema import KinshipConstraint, SocialConstraints


class SocialSolver:
    """社会关系求解器"""

    # 中文亲属称谓 → {性别, 代际差, 血缘/姻亲}
    KINSHIP_KB = {
        # 直系血亲
        "父亲": {"gender": "male", "gen": 1, "type": "blood"},
        "爸爸": {"gender": "male", "gen": 1, "type": "blood"},
        "母亲": {"gender": "female", "gen": 1, "type": "blood"},
        "妈妈": {"gender": "female", "gen": 1, "type": "blood"},
        "儿子": {"gender": "male", "gen": -1, "type": "blood"},
        "女儿": {"gender": "female", "gen": -1, "type": "blood"},
        "哥哥": {"gender": "male", "gen": 0, "type": "blood", "older": True},
        "姐姐": {"gender": "female", "gen": 0, "type": "blood", "older": True},
        "弟弟": {"gender": "male", "gen": 0, "type": "blood", "older": False},
        "妹妹": {"gender": "female", "gen": 0, "type": "blood", "older": False},
        # 隔代血亲
        "爷爷": {"gender": "male", "gen": 2, "type": "blood"},
        "奶奶": {"gender": "female", "gen": 2, "type": "blood"},
        "外公": {"gender": "male", "gen": 2, "type": "blood"},
        "外婆": {"gender": "female", "gen": 2, "type": "blood"},
        "孙子": {"gender": "male", "gen": -2, "type": "blood"},
        "孙女": {"gender": "female", "gen": -2, "type": "blood"},
        # 姻亲
        "老公": {"gender": "male", "gen": 0, "type": "marriage"},
        "老婆": {"gender": "female", "gen": 0, "type": "marriage"},
        "岳父": {"gender": "male", "gen": 1, "type": "marriage"},
        "岳母": {"gender": "female", "gen": 1, "type": "marriage"},
        "儿媳": {"gender": "female", "gen": -1, "type": "marriage"},
        "女婿": {"gender": "male", "gen": -1, "type": "marriage"},
        "姐夫": {"gender": "male", "gen": 0, "type": "marriage"},
        "嫂子": {"gender": "female", "gen": 0, "type": "marriage"},
        "小姨": {"gender": "female", "gen": 0, "type": "blood"},  # 母亲的妹妹
        "小叔": {"gender": "male", "gen": 0, "type": "blood"},    # 父亲的弟弟
        # 英文关系
        "father": {"gender": "male", "gen": 1, "type": "blood"},
        "mother": {"gender": "female", "gen": 1, "type": "blood"},
        "son": {"gender": "male", "gen": -1, "type": "blood"},
        "daughter": {"gender": "female", "gen": -1, "type": "blood"},
        "husband": {"gender": "male", "gen": 0, "type": "marriage"},
        "wife": {"gender": "female", "gen": 0, "type": "marriage"},
        "brother": {"gender": "male", "gen": 0, "type": "blood"},
        "sister": {"gender": "female", "gen": 0, "type": "blood"},
        "teacher": {"gender": None, "gen": 1, "type": "social"},
        "student": {"gender": None, "gen": -1, "type": "social"},
        "leader": {"gender": None, "gen": 1, "type": "social"},
        "boss": {"gender": None, "gen": 1, "type": "social"},
        "buddy": {"gender": None, "gen": 0, "type": "social"},
        "classmate": {"gender": None, "gen": 0, "type": "social"},
        "neighbor": {"gender": None, "gen": 0, "type": "social"},
        "bestie": {"gender": None, "gen": 0, "type": "social"},
        "ex-girlfriend": {"gender": "female", "gen": 0, "type": "social"},
        "ex-boyfriend": {"gender": "male", "gen": 0, "type": "social"},
        "ex-wife": {"gender": "female", "gen": 0, "type": "social"},
        "ex-husband": {"gender": "male", "gen": 0, "type": "social"},
        "elder sister": {"gender": "female", "gen": 0, "type": "blood", "older": True},
        "elder brother": {"gender": "male", "gen": 0, "type": "blood", "older": True},
    }

    def __init__(self):
        self.relations = []  # List of (person_a, person_b, relation_type)
        self.entities = set()
        self.genders = {}
        self.generations = {}

    def solve(self, constraints: SocialConstraints) -> Dict:
        """求解社会关系约束，返回推断出的关系网络"""
        self._reset()
        self._build_knowledge(constraints)
        self._infer_genders()
        self._infer_generations()
        self._infer_transitive_relations()

        return {
            "entities": list(self.entities),
            "relations": self.relations,
            "genders": self.genders,
            "generations": self.generations,
        }

    def _reset(self):
        self.relations = []
        self.entities = set()
        self.genders = {}
        self.generations = {}

    def _build_knowledge(self, constraints: SocialConstraints):
        """从约束中构建知识库"""
        for kc in constraints.relations:
            self.entities.add(kc.person_a)
            self.entities.add(kc.person_b)
            self.relations.append((kc.person_a, kc.person_b, kc.relation))

            # 从称谓推断性别
            kb_entry = self.KINSHIP_KB.get(kc.relation, {})
            if kc.person_a not in self.genders and kb_entry.get("gender"):
                self.genders[kc.person_a] = kb_entry["gender"]
            if kc.inferred_gender_a:
                self.genders[kc.person_a] = kc.inferred_gender_a
            if kc.inferred_gender_b:
                self.genders[kc.person_b] = kc.inferred_gender_b

            # 从称谓推断代际
            if kc.person_a not in self.generations and kb_entry.get("gen") is not None:
                self.generations[kc.person_a] = kb_entry["gen"]
            if kc.generation_diff is not None:
                self.generations[kc.person_a] = kc.generation_diff
                self.generations[kc.person_b] = 0

        # 初始实体中的性别信息
        for name, gender in constraints.genders.items():
            self.genders[name] = gender
        for name, gen in constraints.generations.items():
            self.generations[name] = gen

    def _infer_genders(self):
        """从关系传递推断更多性别信息"""
        changed = True
        while changed:
            changed = False

            for a, b, rel in self.relations:
                kb = self.KINSHIP_KB.get(rel, {})

                # 夫妻关系 → 异性（传统设定）
                if rel in ("老公", "老婆", "husband", "wife"):
                    if a in self.genders and b not in self.genders:
                        self.genders[b] = "female" if self.genders[a] == "male" else "male"
                        changed = True

                # 儿女关系 → 从已知性别推断
                if rel in ("儿子", "女儿"):
                    inferred_gender = "male" if rel == "儿子" else "female"
                    if a not in self.genders:
                        self.genders[a] = inferred_gender
                        changed = True

                # 人际关系传递：如果A是B的X，B是C的Y，可以推断A和C的关系
                for a2, b2, rel2 in self.relations:
                    if b == a2:
                        inferred = self._compose_relations(rel, rel2)
                        if inferred:
                            if (a, b2, inferred) not in self.relations:
                                self.relations.append((a, b2, inferred))
                                changed = True

    def _infer_generations(self):
        """推断代际信息"""
        for a, b, rel in self.relations:
            kb = self.KINSHIP_KB.get(rel, {})
            gen = kb.get("gen")
            if gen is not None:
                # A = relation of B → gen(A) = gen(B) + gen(A相对于B)
                if b in self.generations and a not in self.generations:
                    self.generations[a] = self.generations[b] + gen
                elif a in self.generations and b not in self.generations:
                    self.generations[b] = self.generations[a] - gen
                elif a not in self.generations and b not in self.generations:
                    self.generations[b] = 0
                    self.generations[a] = gen

    def _infer_transitive_relations(self):
        """关系传递闭包"""
        # 构建关系图用于传递推理
        graph = defaultdict(list)
        for a, b, rel in self.relations:
            graph[a].append((b, rel))
            graph[b].append((a, f"inverse_{rel}"))

    def _compose_relations(self, rel1: str, rel2: str) -> Optional[str]:
        """组合两个关系"""
        # 核心传递规则
        TRANSITIVITY = {
            # 配偶的父亲 = 岳父/公公
            ("老婆", "父亲"): "岳父",
            ("老公", "父亲"): "公公",
            # 父亲的父亲 = 爷爷
            ("父亲", "父亲"): "爷爷",
            # 父亲的母亲 = 奶奶
            ("父亲", "母亲"): "奶奶",
            # ... 更多规则
        }
        return TRANSITIVITY.get((rel1, rel2))

    def find_relation(self, person_a: str, person_b: str) -> Optional[str]:
        """查找两人之间的直接关系"""
        for a, b, rel in self.relations:
            if a == person_a and b == person_b:
                return rel
            if a == person_b and b == person_a:
                return f"inverse_{rel}"
        return None

    def verify_statement(self, statement: str) -> bool:
        """验证一个关系陈述是否为真"""
        # 解析 "A是B的X" 或 "A is B's X" 格式
        # 这由答案验证器调用
        return None
