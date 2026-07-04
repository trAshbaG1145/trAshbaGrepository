"""
答案验证器：将求解器结果与选项逐一比对，选出正确答案

支持三种题型：
1. 填空题 (fill_blank): 选出使句子成立的选项
2. 选择题 (select_correct): 选出正确的选项
3. 选非题 (select_incorrect): 选出不正确的选项
"""
from typing import Dict, List, Tuple, Optional, Any
import re


class AnswerVerifier:
    """答案验证器——将求解器输出映射为题目的选项"""

    def __init__(self):
        pass

    def verify(self, question: str, options: Dict[str, str],
               solution: Dict[str, Any], domain: str = "",
               question_type: str = "") -> List[str]:
        """
        验证每个选项，返回正确的选项字母列表

        Args:
            question: 问题文本
            options: {A: "选项内容", B: ...}
            solution: 求解器输出，格式因领域而异
            domain: 领域类型
            question_type: 题型

        Returns:
            正确选项列表，如 ["A", "C"]
        """
        # 自动检测题型（优先检测incorrect，因为"以下选项中不正确的是____"同时包含____和"不正确"）
        if not question_type:
            if "不正确" in question or "incorrect" in question.lower():
                question_type = "select_incorrect"
            elif "____" in question:
                question_type = "fill_blank"
            else:
                question_type = "select_correct"

        # 根据领域和题型验证
        if question_type == "fill_blank":
            return self._verify_fill_blank(question, options, solution, domain)
        elif question_type == "select_incorrect":
            return self._verify_incorrect(question, options, solution, domain)
        else:
            return self._verify_correct(question, options, solution, domain)

    def _verify_fill_blank(self, question: str, options: Dict[str, str],
                            solution: Dict[str, Any], domain: str) -> List[str]:
        """
        验证填空题：判断每个选项填入空白后是否成立

        例如问题: "4号田中种的是____"
        选项: A:莴苣, B:南瓜, C:胡萝卜, D:花生
        求解结果: {"4号田": "南瓜"} → 答案: B
        """
        correct = []

        for letter, option_text in options.items():
            if self._check_fill_option(question, option_text, solution, domain):
                correct.append(letter)

        return correct

    def _verify_correct(self, question: str, options: Dict[str, str],
                         solution: Dict[str, Any], domain: str) -> List[str]:
        """验证选择题：判断哪些选项的陈述为真"""
        correct = []

        for letter, option_text in options.items():
            if self._check_statement(option_text, solution, domain):
                correct.append(letter)

        return correct

    def _verify_incorrect(self, question: str, options: Dict[str, str],
                           solution: Dict[str, Any], domain: str) -> List[str]:
        """验证选非题：判断哪些选项的陈述为假"""
        incorrect = []

        for letter, option_text in options.items():
            if not self._check_statement(option_text, solution, domain):
                incorrect.append(letter)

        return incorrect

    def _check_fill_option(self, question: str, option_text: str,
                            solution: Dict[str, Any], domain: str) -> bool:
        """检查填空选项是否成立"""
        # 根据领域类型选择验证逻辑
        if domain in ("nature", "space+nature", "time+nature"):
            return self._check_nature_option(question, option_text, solution)
        elif domain in ("space", "space+nature", "space+social"):
            return self._check_space_option(question, option_text, solution)
        elif domain in ("time", "time+nature", "time+social"):
            return self._check_time_option(question, option_text, solution)
        elif domain in ("social", "time+social", "space+social"):
            return self._check_social_option(question, option_text, solution)
        else:
            # 通用检查：在解中搜索选项文本
            return self._generic_check(option_text, solution)

    def _check_statement(self, statement: str, solution: Dict[str, Any],
                          domain: str) -> bool:
        """检查一个陈述是否为真"""
        return self._check_fill_option("", statement, solution, domain)

    # ---- 各领域的选项验证逻辑 ----

    def _check_time_option(self, question: str, option_text: str,
                            solution: Dict[str, Any]) -> bool:
        """验证时间类选项"""
        times = solution.get("times", {})

        if not times:
            return False

        # === 填空题模式处理 ===

        # EN: "____ N days after/before X"
        fill_match_en = re.search(r'____\s+(\d+)\s*days?\s+(after|before)\s+(.+)', question, re.IGNORECASE)
        if not fill_match_en:
            fill_match_en = re.search(r'____\s+(after|before)\s+(.+)', question, re.IGNORECASE)

        if fill_match_en:
            groups = fill_match_en.groups()
            if len(groups) == 3:
                offset_str, direction, ref_event = groups
                offset = int(offset_str)
            else:
                direction, ref_event = groups
                offset = 1  # default
            ref_event = ref_event.strip()
            ref_time = self._find_event_time(ref_event, times)
            if ref_time is not None:
                opt_time = self._find_event_time(option_text, times)
                if opt_time is not None:
                    if direction == "after":
                        return (opt_time - ref_time) == offset
                    else:
                        return (ref_time - opt_time) == offset
                return self._fuzzy_check(option_text, solution)

        # CN: "在A之后N天，____" or "在A的N天之后，____"
        cn_fill_patterns = [
            r'在(.+?)之后(\d+)\s*天\s*[,，]?\s*____',     # 在他看电影之后3天，____
            r'在(.+?)之前(\d+)\s*天\s*[,，]?\s*____',     # 在他看电影之前3天，____
            r'在(.+?)的(\d+)\s*天\s*之后\s*[,，]?\s*____', # 在他看电影的3天之后，____
            r'在(.+?)的(\d+)\s*天\s*之前\s*[,，]?\s*____', # 在他看电影的3天之前，____
        ]
        for pattern in cn_fill_patterns:
            m = re.search(pattern, question)
            if m:
                ref_event = m.group(1).strip()
                offset = int(m.group(2))
                is_after = "之后" in pattern
                ref_time = self._find_event_time(ref_event, times)
                if ref_time is not None:
                    opt_time = self._find_event_time(option_text, times)
                    if opt_time is not None:
                        if is_after:
                            return (opt_time - ref_time) == offset
                        else:
                            return (ref_time - opt_time) == offset
                return self._fuzzy_check(option_text, solution)

        # CN: "____在A之后N天" or "____在A的N天之后"
        cn_fill_patterns2 = [
            r'____\s*在(.+?)之后(\d+)\s*天',
            r'____\s*在(.+?)之前(\d+)\s*天',
            r'____\s*在(.+?)的(\d+)\s*天\s*之后',
            r'____\s*在(.+?)的(\d+)\s*天\s*之前',
        ]
        for pattern in cn_fill_patterns2:
            m = re.search(pattern, question)
            if m:
                ref_event = m.group(1).strip()
                offset = int(m.group(2))
                is_after = "之后" in pattern
                ref_time = self._find_event_time(ref_event, times)
                if ref_time is not None:
                    opt_time = self._find_event_time(option_text, times)
                    if opt_time is not None:
                        if is_after:
                            return (opt_time - ref_time) == offset
                        else:
                            return (ref_time - opt_time) == offset
                return self._fuzzy_check(option_text, solution)

        # 检测中文选择题模式:
        # CN: "在A之后N天，B" → B after A (earlier=A, later=B)
        # EN: "A N days after B" → A after B (earlier=B, later=A)
        time_patterns = [
            # "A 和 B 中间差了N天" / "A和B之间相差N天"
            (r'(.+?)(?:和|与)(.+?)(?:中间|之间)(?:相差|差[了]?)(\d+)\s*天', 'diff', None),
            # "A与B相差N天"
            (r'(.+?)与(.+?)相差(\d+)\s*天', 'diff', None),
            # "gap/difference between A and B is N days" (EN)
            (r'(?:gap|difference)\s+between\s+(.+?)\s+and\s+(.+?)\s+is\s+(\d+)\s*days?', 'diff_en', None),
            # EN: "A N days after B" → later=A, earlier=B
            (r'(.+?)\s+(\d+)\s*days?\s+after\s+(.+)', 'after_en', None),
            # EN: "A N days before B" → earlier=A, later=B
            (r'(.+?)\s+(\d+)\s*days?\s+before\s+(.+)', 'before_en', None),
            # CN: "在A之后N天，B" → B after A (later=B, earlier=A)
            (r'(?:在\s*)?(.+?)\s*之后\s*(\d+)\s*天\s*[,，]?\s*(.+)', 'after_cn', None),
            # CN: "在A之前N天，B" → B before A (earlier=B, later=A)
            (r'(?:在\s*)?(.+?)\s*之前\s*(\d+)\s*天\s*[,，]?\s*(.+)', 'before_cn', None),
            # CN: "A的N天之后，B"
            (r'(.+?)\s*的\s*(\d+)\s*天\s*之后\s*[,，]?\s*(.+)', 'after_cn', None),
        ]

        for pattern, rel_type, _ in time_patterns:
            m = re.search(pattern, option_text, re.IGNORECASE)
            if m:
                if rel_type == 'diff':
                    event1, event2, expected_diff = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
                    actual_diff = self._compute_time_diff(event1, event2, times)
                    return actual_diff is not None and abs(actual_diff) == expected_diff
                elif rel_type == 'diff_en':
                    event1, event2, expected_diff = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
                    actual_diff = self._compute_time_diff(event1, event2, times)
                    return actual_diff is not None and abs(actual_diff) == expected_diff
                elif rel_type == 'after_en':
                    # EN: "A N days after B" → later=A(group1), earlier=B(group3)
                    later_event, offset, earlier_event = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
                    actual_diff = self._compute_time_diff(later_event, earlier_event, times)
                    return actual_diff is not None and actual_diff == offset
                elif rel_type == 'before_en':
                    # EN: "A N days before B" → earlier=A(group1), later=B(group3)
                    earlier_event, offset, later_event = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
                    actual_diff = self._compute_time_diff(later_event, earlier_event, times)
                    return actual_diff is not None and actual_diff == offset
                elif rel_type == 'after_cn':
                    # CN: "在A之后N天，B" → B after A, later=B(group3), earlier=A(group1)
                    earlier_event, offset, later_event = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
                    actual_diff = self._compute_time_diff(later_event, earlier_event, times)
                    return actual_diff is not None and actual_diff == offset
                elif rel_type == 'before_cn':
                    # CN: "在A之前N天，B" → B before A, earlier=B(group3), later=A(group1)
                    later_event, offset, earlier_event = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
                    actual_diff = self._compute_time_diff(later_event, earlier_event, times)
                    return actual_diff is not None and actual_diff == offset

        # 检查是否是绝对日期陈述: "在星期X，他Y"
        abs_match = re.search(r'在\s*(星期[一二三四五六日天]|周[一二三四五六日天]|周[一二三四五六日天])\s*[,，]?\s*(.+)', option_text)
        if abs_match:
            weekday = abs_match.group(1)
            event = abs_match.group(2).strip()
            # 从times中找这个时间的星期几
            opt_time_val = self._find_event_time(event, times)
            if opt_time_val is not None:
                weekday_idx = self._parse_weekday(weekday)
                if weekday_idx is not None:
                    return (opt_time_val % 7) == weekday_idx

        # 检查简单事件陈述: "他打羽毛球" 等
        event = option_text.strip().rstrip('。，.')
        if event in times:
            return True

        # 模糊匹配：在时间解中搜索
        return self._generic_check(option_text, solution)

    def _find_event_time(self, event_desc: str, times: Dict[str, int]) -> Optional[int]:
        """在times字典中查找事件的时间（支持模糊匹配和代词消解）"""
        event_desc = event_desc.strip().rstrip('。，,.!;； ')

        # 精确匹配
        if event_desc in times:
            return times[event_desc]

        # 去掉代词前缀后匹配：他跑步→跑步, Jack plays badminton→plays badminton
        clean_desc = self._strip_subject_prefix(event_desc)
        if clean_desc in times:
            return times[clean_desc]

        # times key去掉代词后与option匹配
        for event_key, time_val in times.items():
            clean_key = self._strip_subject_prefix(event_key)
            if clean_key == clean_desc:
                return time_val
            if clean_key and clean_desc and (clean_key in clean_desc or clean_desc in clean_key):
                return time_val

        # 包含匹配（双向）
        for event_key, time_val in times.items():
            if event_key in event_desc or event_desc in event_key:
                return time_val

        return None

    @staticmethod
    def _strip_subject_prefix(text: str) -> str:
        """去除事件名中的主语代词/人名前缀"""
        # 中文代词
        text = re.sub(r'^(他|她|它|他们|她们|它们|其|小明|小红)', '', text)
        # 英文人名/代词前缀 (He/She/They/Jack/Mary...+空格)
        text = re.sub(
            r'^(He|She|They|It|Jack|Mary|John|Tom|Alice|Bob|David|Sarah|James|Linda|'
            r'Michael|Patricia|Robert|Jennifer|William|Elizabeth|Richard|Barbara|Joseph|'
            r'Susan|Thomas|Jessica|Charles|Karen|Christopher|Nancy|Daniel|Lisa|Matthew|'
            r'Betty|Anthony|Margaret|Mark|Sandra|Donald|Ashley|Steven|Kimberly|Paul|'
            r'Emily|Andrew|Donna|Kenneth|Michelle|Joshua|Carol|Kevin|Amanda|Brian|'
            r'Melissa|George|Deborah|Timothy|Stephanie|Ronald|Rebecca|Jason|Sharon|'
            r'Edward|Laura|Jeffrey|Cynthia|Ryan|Kathleen|Jacob|Amy|Gary|Shirley|'
            r'Nicholas|Angela|Eric|Anna|Jonathan|Brenda|Stephen|Pamela|Larry|Nicole|'
            r'Justin|Samantha|Scott|Katherine|Brandon|Emma|Benjamin|Helen|Samuel|'
            r'Christine|Gregory|Sara|Frank|Rachel|Raymond|Carolyn|Patrick|Janet|'
            r'Alexander|Maria|Brian|Nora|Grace|Miranda|Curtis|Violet|Andrew|Wayne)\s+',
            '', text)
        return text.strip()

    def _parse_weekday(self, weekday_str: str) -> Optional[int]:
        """解析星期字符串为0-6的索引"""
        weekdays = {
            '周一': 0, '周二': 1, '周三': 2, '周四': 3, '周五': 4, '周六': 5, '周日': 6,
            '星期一': 0, '星期二': 1, '星期三': 2, '星期四': 3, '星期五': 4, '星期六': 5, '星期日': 6,
            '星期天': 6, '周天': 6,
        }
        return weekdays.get(weekday_str)

    def _check_space_option(self, question: str, option_text: str,
                             solution: Dict[str, Any]) -> bool:
        """验证空间类选项"""
        positions = solution.get("positions", {})

        # "X与Y不同侧" → X和Y不在同一列
        diff_side = re.search(r'(\S+?)与(\S+?)不同侧', option_text)
        if diff_side:
            a, b = diff_side.group(1), diff_side.group(2)
            if a in positions and b in positions:
                return positions[a][1] != positions[b][1]

        # "X是Y的右邻/左邻"
        adjacent = re.search(r'(\S+?)是(\S+?)的(左|右)邻', option_text)
        if adjacent:
            a, b, direction = adjacent.group(1), adjacent.group(2), adjacent.group(3)
            if a in positions and b in positions:
                ra, ca = positions[a]
                rb, cb = positions[b]
                if direction == "左":
                    return ra == rb and cb - ca == 1
                else:
                    return ra == rb and ca - cb == 1

        return self._generic_check(option_text, solution)

    def _check_social_option(self, question: str, option_text: str,
                              solution: Dict[str, Any]) -> bool:
        """验证社会关系选项"""
        # "A是B的C" 格式
        match = re.search(r'(\S+?)是(\S+?)的(\S+)', option_text)
        if match:
            a, b, rel = match.group(1), match.group(2), match.group(3)
            relations = solution.get("relations", [])
            for ra, rb, rrel in relations:
                if ra == a and rb == b and rrel == rel:
                    return True
            return False

        return self._generic_check(option_text, solution)

    def _check_nature_option(self, question: str, option_text: str,
                              solution: Dict[str, Any]) -> bool:
        """验证自然常识选项"""
        assignment = solution.get("assignment", {})

        # "X号田种的是Y" → 检查赋值
        match = re.search(r'(\S+号[田格位])\S*种的?是(\S+)', option_text)
        if match:
            pos, entity = match.group(1), match.group(2)
            return assignment.get(pos) == entity

        # "X is on photo No.Y"
        match = re.search(r'(\S+)\s+is\s+on\s+photo\s+No\.(\d+)', option_text, re.IGNORECASE)
        if match:
            entity, photo_num = match.group(1), match.group(2)
            return assignment.get(f"photo No.{photo_num}") == entity

        # 处理填空题：选项是实体名，问题是"N号田中种的是____"
        # 从问题中提取位置
        pos_match = re.search(r'(\d+号[田格位])', question)
        if pos_match:
            pos = pos_match.group(1)
            entity = option_text.strip().rstrip('。，.')
            if assignment and pos in assignment:
                return assignment[pos] == entity

        return self._generic_check(option_text, solution)

    def _compute_time_diff(self, event_a: str, event_b: str,
                            times: Dict[str, int]) -> Optional[int]:
        """计算两个事件的时间差"""
        # 精确匹配
        if event_a in times and event_b in times:
            return times[event_a] - times[event_b]

        # 模糊匹配（忽略前后缀，如标点）
        clean_a = event_a.strip().rstrip('。，,.')
        clean_b = event_b.strip().rstrip('。，,.')
        if clean_a in times and clean_b in times:
            return times[clean_a] - times[clean_b]

        # 包含匹配
        for key_a in times:
            if key_a in event_a or event_a in key_a:
                for key_b in times:
                    if key_b in event_b or event_b in key_b:
                        return times[key_a] - times[key_b]

        return None

    def _generic_check(self, option_text: str, solution: Dict[str, Any]) -> bool:
        """通用检查：在解中搜索关键词（改进版：token级匹配）"""
        # 将解转为字符串
        sol_str = str(solution).lower()
        opt_str = option_text.lower().strip().rstrip('。，,.!;； ')

        # 精确子串匹配
        if opt_str in sol_str:
            return True

        # Token级匹配：选项中的关键词是否出现在解中
        # 提取选项中的关键实体词（去除常见停用词和虚词）
        key_tokens = self._extract_key_tokens(option_text)
        if key_tokens:
            match_count = sum(1 for t in key_tokens if t.lower() in sol_str)
            # 如果大部分关键词匹配，认为是True
            if len(key_tokens) >= 2 and match_count >= len(key_tokens) * 0.6:
                return True
            elif len(key_tokens) == 1 and match_count == 1:
                return True

        return False

    @staticmethod
    def _extract_key_tokens(text: str) -> List[str]:
        """从选项文本中提取关键词"""
        # 去除标点和常见停用词
        stop_words = {
            '的', '了', '在', '是', '有', '和', '与', '或', '不', '也', '都', '就',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'between', 'and', 'but', 'or', 'not', 'no', 'nor',
            'there', 'here', 'then', 'than', 'that', 'this', 'these', 'those',
            '他', '她', '它', '他们', 'he', 'she', 'it', 'they', 'his', 'her', 'its',
            'their', 'him', 'them', 'we', 'you', 'i', 'me', 'us', 'my', 'your', 'our',
            '怎么样', '什么', '哪些', '哪', '哪个', '谁', 'how', 'what', 'which', 'who', 'whose',
            '以上', '下列', '以下', '其中', '所有', '每个', '各自', '分别',
        }
        # 分词（简单按空格和常见分隔符）
        tokens = re.split(r'[\s,，、。；;：:？！!?()（）""''""\[\]【】]+', text)
        key_tokens = [t for t in tokens if t and len(t) >= 2 and t.lower() not in stop_words]
        return key_tokens
