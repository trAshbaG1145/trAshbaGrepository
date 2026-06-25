from openai import OpenAI


class CareerTimelineAnalyzer:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def analyze_career_timeline(self, career_name):
        # 构造更详细的提示模板
        prompt = f"""请分析"{career_name}"这个行业或职业的未来发展时间线，要求包含以下内容：
        1. 时间阶段（格式：#### **阶段名称（时间范围）**）
        2. 背景（使用- **背景**：开头）
        3. 关键事件（使用- **关键事件**：开头，分点列出）
        4. 技术要求（使用- **技术要求**：开头，分点列出）
        输出格式要求：Markdown，使用三级标题，时间阶段用####，子项用-加粗
        示例：
        #### **阶段名称（时间范围）**
        - **背景**：...
        - **关键事件**：
          - 事件1
          - 事件2
        - **技术要求**：
          - 技能1
          - 技能2"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
