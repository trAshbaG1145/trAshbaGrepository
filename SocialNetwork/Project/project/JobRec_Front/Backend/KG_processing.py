import re


def process_sentence(sentence):
    """
    智能解析用户输入，动态提取技能和学历信息

    参数:
    sentence (str): 用户输入的自然语言文本

    返回:
    tuple: (技能列表, 学历)
    """
    skill_verbs = ["会", "擅长", "掌握", "熟悉", "了解", "精通"]
    edu_keywords = r"本科|硕士|博士|大专|学士|研究生|大学学历"

    # 清理干扰词（保留逗号、顿号以辅助切割）
    sentence = re.sub(
        r"的|是|有|一些|适合干什么|可以做什么|能做什么|求推荐|[？?]",
        "",
        sentence
    )

    # 提取学历（支持复杂表达如"211本科"）
    education = None
    edu_pattern = re.compile(
        fr"({edu_keywords}+(?:学历|毕业|学位)?)|"  # 正向匹配
        fr"(?:学历[为是])({edu_keywords})"  # 处理"学历为硕士"结构
    )
    if match := edu_pattern.search(sentence):
        education = match.group(1) or match.group(2)

    # 提取技能（支持多种动词和分隔符）
    skill_pattern = re.compile(
        fr"(?:{'|'.join(skill_verbs)})"  # 匹配技能动词
        r"[\s、，；,]*"  # 跳过动词后的分隔符
        r"([^。；！!?？]+)"  # 捕获直到句子结束符
    )
    skills = []
    for match in skill_pattern.finditer(sentence):
        # 多级分割处理不同分隔符
        parts = re.split(r"[，、；,;\s]+", match.group(1))
        skills.extend([p.strip() for p in parts if p.strip()])

    # 后处理：合并简写并去重
    skill_alias = {"js": "JavaScript", "py": "Python"}
    skills = [skill_alias.get(s.lower(), s) for s in skills]

    return list(set(skills)), education


if __name__ == "__main__":
    test_cases = [
        "硕士学历，擅长数据分析，机器学习，求推荐",
        "会py，js，有三年工作经验",
    ]

    for text in test_cases:
        skills, edu = process_sentence(text)
        print(f"输入：{text}")
        print(f"技能：{skills}\n学历：{edu}\n{'-' * 30}")
