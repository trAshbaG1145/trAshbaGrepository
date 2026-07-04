"""
空间推理求解器

核心算法：约束满足问题(CSP) + 回溯搜索
- 构建空间结构（3×2网格/圆形/线性）
- 逐条应用约束缩小可能位置
- 回溯搜索找到所有满足约束的布局
"""
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy
from itertools import permutations
from src.constraint_schema import (
    SpaceConstraints, SpatialStructure,
    PositionConstraint, SpatialRelationConstraint
)


class SpatialSolver:
    """空间约束求解器"""

    def __init__(self, structure: SpatialStructure = SpatialStructure.GRID_3X2,
                 rows: int = 3, cols: int = 2,
                 col_labels: List[str] = None):
        self.structure = structure
        self.rows = rows
        self.cols = cols
        self.col_labels = col_labels or ["东", "西"]

    def solve(self, constraints: SpaceConstraints) -> Dict[str, Tuple[int, int]]:
        """
        求解空间约束，返回实体→(行, 列)的映射
        行: 0=顶层, 1=中层, 2=底层 (或根据坐标系)
        列: 0=东/左, 1=西/右
        """
        self.rows = constraints.rows
        self.cols = constraints.cols
        self.col_labels = constraints.col_labels
        self.structure = constraints.structure

        entities = list(constraints.entities)
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]

        if len(entities) > len(all_positions):
            return {}  # 实体数超过位置数，无法求解

        # 构建约束检查函数
        constraint_checks = []
        for pos_con in constraints.positions:
            constraint_checks.append(self._build_position_check(pos_con))
        for rel_con in constraints.relations:
            constraint_checks.append(self._build_relation_check(rel_con))

        # 回溯搜索
        solutions = []
        self._backtrack(entities, all_positions, {}, constraint_checks, solutions)

        if len(solutions) == 1:
            return solutions[0]
        elif len(solutions) > 1:
            # 多解：返回所有解（后续需要LLM判断）
            return solutions[0]  # 先返回第一个
        return {}

    def _build_position_check(self, constraint: PositionConstraint):
        """构建位置约束的检查函数"""
        entity = constraint.entity
        target_row = constraint.row
        target_col = constraint.col

        def check(assignment: Dict[str, Tuple[int, int]]) -> bool:
            if entity not in assignment:
                return True
            r, c = assignment[entity]
            if target_row is not None and r != target_row:
                return False
            if target_col is not None:
                col_idx = self.col_labels.index(target_col) if target_col in self.col_labels else None
                if col_idx is not None and c != col_idx:
                    return False
            return True

        return check

    def _build_relation_check(self, constraint: SpatialRelationConstraint):
        """构建空间关系约束的检查函数"""
        entity_a = constraint.entity_a
        entity_b = constraint.entity_b
        relation = constraint.relation
        gap = constraint.gap

        def check(assignment: Dict[str, Tuple[int, int]]) -> bool:
            if entity_a not in assignment or entity_b not in assignment:
                return True
            r1, c1 = assignment[entity_a]
            r2, c2 = assignment[entity_b]

            if relation == "above":
                # A在B正上方，且可能隔N层
                if c1 != c2:
                    return False
                return r2 - r1 == gap + 1

            elif relation == "below":
                if c1 != c2:
                    return False
                return r1 - r2 == gap + 1

            elif relation == "adjacent_left":
                # A是B的左邻（同层相邻，A在左/B在右）
                return r1 == r2 and c2 - c1 == 1

            elif relation == "adjacent_right":
                # A是B的右邻
                return r1 == r2 and c1 - c2 == 1

            elif relation == "same_row":
                return r1 == r2

            elif relation == "same_col":
                return c1 == c2

            elif relation == "different_col":
                return c1 != c2

            elif relation == "different_row":
                return r1 != r2

            elif relation == "above_or_below":
                # A在B的上方或下方（可能不同列），隔N层
                return abs(r1 - r2) == gap + 1

            elif relation == "diagonal":
                return abs(r1 - r2) == 1 and abs(c1 - c2) == 1

            # 圆形排列的关系
            elif relation == "clockwise":
                # 在圆形中，顺时针方向
                return True  # 简化：圆中位置由索引确定

            elif relation == "counterclockwise":
                return True

            return True

        return check

    def _backtrack(self, entities: List[str], positions: List[Tuple[int, int]],
                   assignment: Dict[str, Tuple[int, int]],
                   constraints: List[callable],
                   solutions: List[Dict[str, Tuple[int, int]]],
                   max_solutions: int = 10):
        """回溯搜索满足所有约束的排列"""
        if len(assignment) == len(entities):
            solutions.append(dict(assignment))
            return

        if len(solutions) >= max_solutions:
            return

        current_idx = len(assignment)
        entity = entities[current_idx]

        for pos in positions:
            if pos in assignment.values():
                continue
            assignment[entity] = pos

            # 检查所有约束
            if all(check(assignment) for check in constraints):
                self._backtrack(entities, positions, assignment,
                               constraints, solutions, max_solutions)

            del assignment[entity]

    def get_position_description(self, entity: str,
                                  assignment: Dict[str, Tuple[int, int]]) -> str:
        """将位置转化为人类可读的描述"""
        if entity not in assignment:
            return "未知"
        r, c = assignment[entity]
        layer_names = ["上层", "中层", "底层", "一层", "二层", "三层"]
        side_names = self.col_labels
        layer = layer_names[r] if r < len(layer_names) else f"第{r+1}层"
        side = side_names[c] if c < len(side_names) else f"第{c+1}列"
        return f"{layer}{side}"
