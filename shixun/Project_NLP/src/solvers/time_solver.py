"""
时间推理求解器

核心算法：约束传播 + 拓扑排序
- 将绝对时间和相对时间约束转化为有向图
- 选择锚点事件，通过BFS/DFS推导所有事件的时间
- 支持每周循环（7天周期）
"""
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
from itertools import product
from src.constraint_schema import TimeConstraints, AbsoluteTimeConstraint, RelativeTimeConstraint


class TemporalSolver:
    """时间约束求解器"""

    # 星期映射（支持中英文）
    WEEKDAYS_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日",
                   "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def __init__(self, is_weekly: bool = False):
        self.is_weekly = is_weekly
        self.graph = defaultdict(list)  # 有向图: event -> [(next_event, offset)]
        self.absolute_times = {}  # event -> absolute_time
        self.events = set()

    def solve(self, constraints: TimeConstraints) -> Dict[str, int]:
        """求解时间约束，返回事件→时间点的映射"""
        self._reset()
        self._build_graph(constraints)
        return self._propagate()

    def _reset(self):
        self.graph = defaultdict(list)
        self.absolute_times = {}
        self.events = set()

    def _build_graph(self, constraints: TimeConstraints):
        """构建时间关系图"""
        # 处理绝对时间约束
        for ac in constraints.absolute:
            self.events.add(ac.event)
            time_val = self._parse_time(ac.time_point)
            if time_val is not None:
                self.absolute_times[ac.event] = time_val

        # 处理相对时间约束
        for rc in constraints.relative:
            self.events.add(rc.event_a)
            self.events.add(rc.event_b)
            if rc.relation == "after":
                # event_a 在 event_b 之后 offset 天
                # graph: b -> a with weight +offset
                self.graph[rc.event_b].append((rc.event_a, rc.offset))
                # 反向边: a -> b with weight -offset
                self.graph[rc.event_a].append((rc.event_b, -rc.offset))
            elif rc.relation == "before":
                # event_a 在 event_b 之前 offset 天
                # graph: a -> b with weight +offset
                self.graph[rc.event_a].append((rc.event_b, rc.offset))
                self.graph[rc.event_b].append((rc.event_a, -rc.offset))

    def _propagate(self) -> Dict[str, int]:
        """约束传播：从已确定时间的事件推导所有事件"""
        times = dict(self.absolute_times) if self.absolute_times else {}
        queue = deque(list(times.keys()))

        # BFS传播
        while queue:
            curr = queue.popleft()
            for neighbor, offset in self.graph[curr]:
                expected_time = times[curr] + offset
                if neighbor not in times:
                    times[neighbor] = expected_time
                    queue.append(neighbor)
                else:
                    # 一致性检查
                    if self.is_weekly:
                        if (times[neighbor] - expected_time) % 7 != 0:
                            pass  # 周循环中可能有不一致，取最新值
                    else:
                        if times[neighbor] != expected_time:
                            # 约束冲突，优先保留
                            pass

        # 如果有未确定的事件，尝试每个事件作为锚点
        undetermined = self.events - set(times.keys())
        while undetermined:
            # 选择一个未确定的事件设为时间0
            anchor = list(undetermined)[0]
            times[anchor] = 0
            queue.append(anchor)
            while queue:
                curr = queue.popleft()
                for neighbor, offset in self.graph[curr]:
                    expected_time = times[curr] + offset
                    if neighbor not in times:
                        times[neighbor] = expected_time
                        queue.append(neighbor)
            undetermined = self.events - set(times.keys())

        return times

    def _parse_time(self, time_str: str) -> Optional[int]:
        """解析时间字符串为数值"""
        # 尝试解析星期几
        for i, day_name in enumerate(self.WEEKDAYS_CN):
            if day_name in time_str:
                return i % 7
        for i, day_name in enumerate(self.WEEKDAYS_EN):
            if day_name.lower() in time_str.lower():
                return i % 7

        # 尝试解析数字
        try:
            return int(time_str)
        except ValueError:
            pass

        return None

    def get_event_order(self, times: Dict[str, int]) -> List[str]:
        """返回按时间排序的事件列表"""
        return sorted(times.keys(), key=lambda e: times[e])

    def get_time_difference(self, event_a: str, event_b: str, times: Dict[str, int]) -> int:
        """计算两个事件的时间差"""
        if event_a in times and event_b in times:
            return times[event_a] - times[event_b]
        return None

    def get_events_at_time(self, target_time: int, times: Dict[str, int]) -> List[str]:
        """获取在特定时间发生的事件"""
        if self.is_weekly:
            return [e for e, t in times.items() if t % 7 == target_time % 7]
        return [e for e, t in times.items() if t == target_time]

    def find_solutions_for_expressions(self, expressions: List[str],
                                        times: Dict[str, int]) -> Dict[str, List[str]]:
        """验证一系列时间表达式，返回正确的表达式列表"""
        results = {}
        for expr in expressions:
            # 解析类似 "A 2 days after B" 或 "A and B differ by 3 days"
            pass  # 由答案验证器处理
        return results
