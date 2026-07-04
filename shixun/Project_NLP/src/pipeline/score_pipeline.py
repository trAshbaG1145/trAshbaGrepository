"""
SCoRE端到端推理Pipeline

整合约束解析器 + 符号求解器 + 答案验证器
实现从自然语言输入到最终答案的完整推理流程
"""
import json
import sys
from typing import Dict, List

from src.solvers.time_solver import TemporalSolver
from src.solvers.space_solver import SpatialSolver
from src.solvers.social_solver import SocialSolver
from src.solvers.nature_solver import NatureSolver
from src.solvers.fusion_solver import FusionSolver
from src.pipeline.answer_verifier import AnswerVerifier
from src.parser.constraint_parser import TemplateConstraintParser


class SCOREPipeline:
    """SCoRE评测端到端推理Pipeline"""

    def __init__(self, parser=None, use_fusion: bool = True):
        """
        Args:
            parser: 约束解析器（默认使用模板解析器）
            use_fusion: 是否使用融合求解器
        """
        self.parser = parser or TemplateConstraintParser()
        self.use_fusion = use_fusion

        # 初始化求解器
        self.solvers = {
            "time": TemporalSolver(is_weekly=True),
            "space": SpatialSolver(),
            "social": SocialSolver(),
            "nature": NatureSolver(),
        }
        self.fusion_solver = FusionSolver() if use_fusion else None
        self.verifier = AnswerVerifier()

        # 统计信息
        self.stats = {
            "total": 0,
            "correct": 0,
            "parse_failures": 0,
            "solve_failures": 0,
            "by_domain": {},
        }

    def solve_single(self, sample: Dict) -> List[str]:
        """
        求解单道题目

        Args:
            sample: {"id", "domain", "language", "text", "question", "options"}

        Returns:
            答案列表，如 ["A", "B"]
        """
        self.stats["total"] += 1
        domain = sample["domain"]
        language = sample.get("language", "cn")
        text = sample["text"]
        question = sample["question"]
        options = sample["options"]

        # 初始化领域统计
        if domain not in self.stats["by_domain"]:
            self.stats["by_domain"][domain] = {"total": 0, "correct": 0}
        self.stats["by_domain"][domain]["total"] += 1

        # Step 1: 解析约束
        try:
            constraints = self.parser.parse(text, domain, question, language)
        except Exception as e:
            print(f"[Pipeline] Parse failed for {sample.get('id', '?')}: {e}")
            self.stats["parse_failures"] += 1
            return []  # 解析失败，返回空列表

        # Step 2: 求解约束
        try:
            if "+" in domain and self.fusion_solver:
                fusion_result = self.fusion_solver.solve(constraints)
                solution = self._extract_solution(fusion_result, domain)
            else:
                base_domain = domain.split("+")[0]
                solver = self.solvers.get(base_domain)
                if solver and constraints:
                    solution = self._solve_single_domain(solver, constraints, base_domain)
                else:
                    solution = {}
        except Exception as e:
            print(f"[Pipeline] Solve failed for {sample.get('id', '?')}: {e}")
            self.stats["solve_failures"] += 1
            solution = {}

        # Step 3: 验证选项
        answers = self.verifier.verify(question, options, solution, domain)

        return answers

    def _solve_single_domain(self, solver, constraints, domain: str) -> Dict:
        """求解单个领域的约束"""
        if domain == "time" and constraints.time:
            times = solver.solve(constraints.time)
            return {"times": times, "entities": constraints.time.entities}
        elif domain == "space" and constraints.space:
            positions = solver.solve(constraints.space)
            return {"positions": positions, "entities": constraints.space.entities}
        elif domain == "social" and constraints.social:
            return solver.solve(constraints.social)
        elif domain == "nature" and constraints.nature:
            assignment = solver.solve(constraints.nature)
            return {"assignment": assignment, "entities": constraints.nature.entities}
        return {}

    def _extract_solution(self, fusion_result: Dict, domain: str) -> Dict:
        """从融合求解器结果中提取综合解"""
        combined = fusion_result.get("combined", {})
        results = combined.get("results", {})

        # 合并所有子领域的结果
        merged = {}
        for sub_domain, sub_result in results.items():
            merged.update(sub_result)
        return merged

    def evaluate(self, samples: List[Dict]) -> Dict:
        """批量评估并返回准确率"""
        self.stats = {"total": 0, "correct": 0, "parse_failures": 0,
                       "solve_failures": 0, "by_domain": {}}

        results = []
        for sample in samples:
            predicted = self.solve_single(sample)
            actual = sample.get("answers", [])

            is_correct = set(predicted) == set(actual)
            if is_correct:
                self.stats["correct"] += 1
                if sample["domain"] in self.stats["by_domain"]:
                    self.stats["by_domain"][sample["domain"]]["correct"] += 1

            results.append({
                "id": sample["id"],
                "domain": sample["domain"],
                "predicted": predicted,
                "actual": actual,
                "correct": is_correct,
            })

        # 计算准确率
        stats = dict(self.stats)
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for domain, ds in stats["by_domain"].items():
            ds["accuracy"] = ds["correct"] / ds["total"] if ds["total"] > 0 else 0

        return {"results": results, "stats": stats}

    def generate_submission(self, test_samples: List[Dict],
                             output_path: str):
        """为测试集生成提交文件"""
        submissions = []
        for sample in test_samples:
            predicted = self.solve_single(sample)
            submissions.append({
                "id": sample["id"],
                "answers": predicted,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submissions, f, ensure_ascii=False, indent=2)

        print(f"Submission saved to {output_path}")
        print(f"Total predictions: {len(submissions)}")
        print(f"Non-empty predictions: {sum(1 for s in submissions if s['answers'])}")

        return submissions

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("Pipeline Statistics")
        print("=" * 50)
        print(f"Total:      {self.stats['total']}")
        print(f"Correct:    {self.stats['correct']}")
        print(f"Accuracy:   {self.stats['correct']/max(self.stats['total'],1)*100:.1f}%")
        print(f"Parse Err:  {self.stats['parse_failures']}")
        print(f"Solve Err:  {self.stats['solve_failures']}")
        print("\nBy Domain:")
        for domain, ds in sorted(self.stats["by_domain"].items()):
            acc = ds["correct"] / max(ds["total"], 1) * 100
            print(f"  {domain:20s}: {ds['correct']:4d}/{ds['total']:4d} ({acc:5.1f}%)")
