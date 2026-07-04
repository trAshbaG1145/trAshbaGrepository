"""
融合推理求解器

核心算法：分阶段求解 + 信息传递
- 识别涉及的子领域（如 time+social）
- 先求解一个领域，将结果注入另一个领域
- 例如：先确定社交关系 → 用社交称谓指代人 → 求解时间约束
"""
from typing import Dict, List, Tuple, Optional, Any
from src.constraint_schema import StructuredConstraints
from .time_solver import TemporalSolver
from .space_solver import SpatialSolver
from .social_solver import SocialSolver
from .nature_solver import NatureSolver


class FusionSolver:
    """融合域求解器，编排子领域求解器"""

    # 融合域求解顺序定义
    FUSION_ORDER = {
        "time+social": ["social", "time"],     # 先解社交关系，再解时间
        "time+nature": ["nature", "time"],      # 先解自然属性，再解时间
        "space+social": ["social", "space"],    # 先解社交关系，再解空间
        "space+nature": ["nature", "space"],    # 先解自然属性，再解空间
    }

    def __init__(self):
        self.solvers = {
            "time": TemporalSolver(is_weekly=True),
            "space": SpatialSolver(),
            "social": SocialSolver(),
            "nature": NatureSolver(),
        }

    def solve(self, constraints: StructuredConstraints) -> Dict[str, Any]:
        """
        求解融合域约束
        返回综合结果
        """
        domain = constraints.domain

        if "+" not in domain:
            # 单域直接求解
            return self._solve_single(domain, constraints)

        # 融合域分阶段求解
        solve_order = self.FUSION_ORDER.get(domain, domain.split("+"))
        results = {}

        for i, sub_domain in enumerate(solve_order):
            sub_result = self._solve_single(sub_domain, constraints)

            if sub_result:
                results[sub_domain] = sub_result
                # 将结果注入到约束中供下一阶段使用
                self._inject_results(constraints, sub_domain, sub_result)

        return {
            "domain": domain,
            "results": results,
            "combined": self._combine_results(results, domain),
        }

    def _solve_single(self, domain: str,
                       constraints: StructuredConstraints) -> Optional[Dict]:
        """求解单个领域"""
        solver = self.solvers.get(domain)
        if not solver:
            return None

        try:
            if domain == "time" and constraints.time:
                return {
                    "times": solver.solve(constraints.time),
                    "events": constraints.time.entities,
                }
            elif domain == "space" and constraints.space:
                return {
                    "positions": solver.solve(constraints.space),
                    "entities": constraints.space.entities,
                }
            elif domain == "social" and constraints.social:
                return solver.solve(constraints.social)
            elif domain == "nature" and constraints.nature:
                return {
                    "assignment": solver.solve(constraints.nature),
                    "entities": constraints.nature.entities,
                    "positions": constraints.nature.positions,
                }
        except Exception as e:
            print(f"[FusionSolver] Error solving {domain}: {e}")
            return None

        return None

    def _inject_results(self, constraints: StructuredConstraints,
                         solved_domain: str, result: Dict):
        """
        将已求解领域的结果注入到约束中
        例如：social求解完 → 将人名→关系映射注入到time约束的实体引用中
        """
        if solved_domain == "social":
            # 社交求解完：将社交称谓替换为具体人名
            self._inject_social_to_others(constraints, result)
        elif solved_domain == "nature":
            # 自然求解完：将属性位置映射为具体实体
            self._inject_nature_to_others(constraints, result)

    def _inject_social_to_others(self, constraints: StructuredConstraints,
                                  social_result: Dict):
        """将社交关系结果注入到其他领域的约束"""
        relations = social_result.get("relations", [])
        genders = social_result.get("genders", {})

        # 构建称谓映射：例如 "Maria Miller's leader" → "Brian Lopez"
        reference_map = self._build_reference_map(relations, genders)

        # 注入到时间约束
        if constraints.time:
            self._resolve_references_in_time(constraints.time, reference_map)

        # 注入到空间约束
        if constraints.space:
            self._resolve_references_in_space(constraints.space, reference_map)

    def _inject_nature_to_others(self, constraints: StructuredConstraints,
                                  nature_result: Dict):
        """将自然属性结果注入到其他领域的约束"""
        assignment = nature_result.get("assignment", {})

        # 构建属性映射
        property_map = {}
        for position, entity in assignment.items():
            property_map[position] = entity

        # 注入到时间约束中
        if constraints.time:
            self._resolve_nature_references_in_time(constraints.time, property_map)

    def _build_reference_map(self, relations: List[Tuple], genders: Dict) -> Dict[str, str]:
        """
        构建引用映射表
        例如 "Maria Miller's leader" → "Brian Lopez"
        """
        ref_map = {}

        # 从已知关系构建：person + relation → target
        for a, b, rel in relations:
            # "A is B's X" → "B's X" refers to A
            ref_key = f"{b}'s {rel}"
            ref_map[ref_key] = a
            ref_map[f"{b}的{rel}"] = a

        return ref_map

    def _resolve_references_in_time(self, time_constraints, reference_map: Dict):
        """解析时间约束中的社交引用"""
        # 替换绝对时间约束中的引用
        for ac in time_constraints.absolute:
            if ac.event in reference_map:
                ac.event = reference_map[ac.event]

        # 替换相对时间约束中的引用
        for rc in time_constraints.relative:
            if rc.event_a in reference_map:
                rc.event_a = reference_map[rc.event_a]
            if rc.event_b in reference_map:
                rc.event_b = reference_map[rc.event_b]

    def _resolve_references_in_space(self, space_constraints, reference_map: Dict):
        """解析空间约束中的社交引用"""
        for pc in space_constraints.positions:
            if pc.entity in reference_map:
                pc.entity = reference_map[pc.entity]
        for rc in space_constraints.relations:
            if rc.entity_a in reference_map:
                rc.entity_a = reference_map[rc.entity_a]
            if rc.entity_b in reference_map:
                rc.entity_b = reference_map[rc.entity_b]

    def _resolve_nature_references_in_time(self, time_constraints,
                                            property_map: Dict[str, str]):
        """解析时间约束中的自然属性引用"""
        for ac in time_constraints.absolute:
            for pos_name, entity_name in property_map.items():
                if pos_name in ac.event:
                    ac.event = ac.event.replace(pos_name, entity_name)

        for rc in time_constraints.relative:
            for pos_name, entity_name in property_map.items():
                if pos_name in rc.event_a:
                    rc.event_a = rc.event_a.replace(pos_name, entity_name)
                if pos_name in rc.event_b:
                    rc.event_b = rc.event_b.replace(pos_name, entity_name)

    def _combine_results(self, results: Dict[str, Any],
                          domain: str) -> Dict[str, Any]:
        """综合多个领域的结果"""
        combined = {"domain": domain, "results": {}}

        for sub_domain, sub_result in results.items():
            if sub_result:
                combined["results"][sub_domain] = sub_result

        return combined
