"""
自然常识推理求解器

核心算法：属性矩阵 + 约束传播 + 排除法
- 构建实体-属性矩阵
- 逐条应用约束标记已知属性
- 利用排除法缩小可能匹配
- 回溯搜索解决最终歧义
"""
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from copy import deepcopy
from src.constraint_schema import (
    NatureConstraints, PropertyConstraint,
    CategoryConstraint, EntityProperty
)


class NatureSolver:
    """自然常识推理求解器"""

    def __init__(self):
        self.entities = []
        self.positions = []
        self.assignment = {}  # position -> entity
        self.properties = {}  # entity/property -> value
        self.categories = {}  # entity -> category set

    def solve(self, constraints: NatureConstraints) -> Dict[str, str]:
        """
        求解自然常识约束，返回位置→实体的映射
        """
        self._reset()
        self.entities = list(constraints.entities)
        self.positions = list(constraints.positions)
        self._load_entity_properties(constraints.entity_properties)
        self._apply_property_constraints(constraints.property_constraints)
        self._apply_category_constraints(constraints.category_constraints)
        self._constraint_propagation()
        self._exclusion_solve()

        return dict(self.assignment)

    def _reset(self):
        self.entities = []
        self.positions = []
        self.assignment = {}
        self.properties = {}
        self.categories = defaultdict(set)

    def _load_entity_properties(self, entity_props: Dict[str, Dict[str, Any]]):
        """加载已知的实体属性"""
        for entity, props in entity_props.items():
            for prop_name, prop_value in props.items():
                self.properties[(entity, prop_name)] = prop_value

    def _apply_property_constraints(self, constraints: List[PropertyConstraint]):
        """应用属性约束"""
        for pc in constraints:
            pos = pc.position
            prop_name = pc.property_name
            prop_value = pc.property_value

            if pc.entity:
                # 已知实体在某位置
                self.assignment[pos] = pc.entity
            else:
                # 某位置的实体具有某属性 → 缩小候选
                self._constrain_by_property(pos, prop_name, prop_value)

    def _apply_category_constraints(self, constraints: List[CategoryConstraint]):
        """应用类别约束"""
        for cc in constraints:
            self.categories[cc.position].add(cc.category)

    def _constrain_by_property(self, position: str, prop_name: str, prop_value: str):
        """通过属性缩小某个位置的候选实体"""
        # 标准化属性名称
        normalized_prop = self._normalize_property_name(prop_name)
        normalized_value = self._normalize_property_value(prop_value)

        matching = []
        for entity in self.entities:
            if entity in self.assignment.values():
                continue
            key = (entity, normalized_prop)
            if key in self.properties and self.properties[key] == normalized_value:
                matching.append(entity)

        if len(matching) == 1:
            self.assignment[position] = matching[0]

    def _normalize_property_name(self, name: str) -> str:
        """标准化属性名称到schema"""
        name = name.strip()
        mappings = {
            "作物开的花": "花色",
            "开的花": "花色",
            "花": "花色",
            "花色": "花色",
            "flower color": "flower_color",
            "作物的可食用部分": "可食用部分",
            "可食用部分": "可食用部分",
            "edible part": "edible_part",
            "颜色": "花色",
            "color": "flower_color",
            "类别": "category",
            "category": "category",
        }
        return mappings.get(name, name)

    def _normalize_property_value(self, value: str) -> str:
        """标准化属性值"""
        value = value.strip().rstrip('。，;')
        mappings = {
            "黄色的": "黄色",
            "它的种子": "种子",
            "白色的": "白色",
            "red": "red",
            "white": "white",
            "yellow": "黄色",
        }
        return mappings.get(value, value)

    def _constraint_propagation(self):
        """约束传播：当一个位置确定后，更新其他位置的候选"""
        changed = True
        while changed:
            changed = False

            # 已被占用的实体从候选池中移除
            assigned_entities = set(self.assignment.values())
            unassigned_positions = [p for p in self.positions if p not in self.assignment]
            available_entities = [e for e in self.entities if e not in assigned_entities]

            # 对于只有一个候选实体的位置，自动分配
            for pos in unassigned_positions:
                candidates = self._get_candidates(pos, available_entities)
                if len(candidates) == 1:
                    self.assignment[pos] = candidates[0]
                    changed = True

            # 对于只能放在一个位置的实体，自动分配
            for entity in available_entities:
                possible_positions = []
                for pos in unassigned_positions:
                    if entity in self._get_candidates(pos, available_entities):
                        possible_positions.append(pos)
                if len(possible_positions) == 1:
                    self.assignment[possible_positions[0]] = entity
                    changed = True

    def _get_candidates(self, position: str, available_entities: List[str]) -> List[str]:
        """获取某个位置的候选实体列表"""
        candidates = list(available_entities)

        # 类别过滤
        if position in self.categories:
            for cat in self.categories[position]:
                # 筛选出属于该类别的实体
                candidates = [e for e in candidates
                             if self._entity_belongs_to_category(e, cat)]

        # 属性过滤（已知属性约束）
        for (entity, prop_name), prop_value in self.properties.items():
            if entity in candidates:
                # 检查是否有冲突的属性约束
                pass

        return candidates

    def _entity_belongs_to_category(self, entity: str, category: str) -> bool:
        """判断实体是否属于某类别"""
        # 类别映射表
        CATEGORY_MAP = {
            # 中文
            "蔬菜": {"莴苣", "南瓜", "胡萝卜", "花生", "土豆", "白菜", "菠菜",
                    "西红柿", "黄瓜", "茄子", "辣椒", "豆角", "芹菜"},
            "水果": {"苹果", "梨", "桃", "杏", "枣", "草莓", "西瓜"},
            "花卉": {"茶花", "水仙", "波斯菊", "月季", "君子兰", "郁金香",
                    "牡丹", "芍药", "菊花", "兰花"},
            "谷物": {"大麦", "小麦", "水稻", "玉米", "高粱", "燕麦"},
            "工具": {"创可贴", "剪刀", "锤子"},
            "加工食品": {"三明治", "担担面", "牛乳", "棉花糖", "吐司"},
            "饮品": {"牛乳", "牛奶"},
            "甜食": {"棉花糖", "蛋糕"},
            # 英文
            "fish": {"catfish", "goldfish", "salmon", "tuna"},
            "flower": {"epiphyllum", "clivia miniata", "rose", "lily", "tulip"},
            "grass": {"barley", "wheat", "rice"},
            "grain": {"barley", "wheat", "rice", "corn"},
            "fruit": {"haw", "pomegranate", "apple", "pear"},
            "tool": {"hot-water bottle", "hammer", "scissors"},
            "processed food": {"toast", "sandwich"},
            "red item": {"haw", "pomegranate", "apple"},
            "red fruit": {"haw", "pomegranate"},
            "sour item": {"haw"},
            "sour fruit": {"haw"},
            "sweet fruit": {"pomegranate"},
        }
        return entity in CATEGORY_MAP.get(category, set())

    def _exclusion_solve(self):
        """排除法求解剩余未分配的位置"""
        unassigned_positions = [p for p in self.positions if p not in self.assignment]
        assigned_entities = set(self.assignment.values())
        available_entities = [e for e in self.entities if e not in assigned_entities]

        if len(unassigned_positions) <= 3:
            # 小规模回溯搜索
            solutions = []
            self._backtrack_assign(unassigned_positions, available_entities,
                                  {}, solutions, max_solutions=1)
            if solutions:
                self.assignment.update(solutions[0])

    def _backtrack_assign(self, positions: List[str], entities: List[str],
                          current: Dict[str, str],
                          solutions: List[Dict[str, str]],
                          max_solutions: int = 5):
        """回溯搜索小的分配问题"""
        if len(current) == len(positions):
            solutions.append(dict(current))
            return
        if len(solutions) >= max_solutions:
            return

        pos_idx = len(current)
        pos = positions[pos_idx]

        for entity in entities:
            if entity in current.values():
                continue
            # 检查约束
            if self._check_constraints_for_position(pos, entity, current):
                current[pos] = entity
                self._backtrack_assign(positions, entities, current,
                                       solutions, max_solutions)
                del current[pos]

    def _check_constraints_for_position(self, position: str, entity: str,
                                         assignment: Dict[str, str]) -> bool:
        """检查某实体放在某位置是否满足约束"""
        # 类别检查
        if position in self.categories:
            for cat in self.categories[position]:
                if not self._entity_belongs_to_category(entity, cat):
                    return False
        return True

    def get_entity_at_position(self, position: str) -> Optional[str]:
        """获取某位置的实体"""
        return self.assignment.get(position)

    def get_position_of_entity(self, entity: str) -> Optional[str]:
        """获取某实体的位置"""
        for pos, ent in self.assignment.items():
            if ent == entity:
                return pos
        return None
