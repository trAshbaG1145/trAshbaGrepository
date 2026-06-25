from openai import OpenAI


class JobRecommender:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def jobRecommender(self, resume):
        # 构造更详细的提示模板
        system_prompt = ("""输出格式要求：Markdown，使用三级标题，职业名称用####，子项用-加粗，匹配度和未来发展行情的子项两到三个，三个概率和应该是100%
        示例：
        #### **职业名称（概率:--%）**
        - **市场状况**：...
        - **匹配度**:
          - 1
          - 2
        - **未来发展行情**:
          - 1
          - 2""")

        user_prompt = f"请分析以下简历文本，给出最适合的三个职业，每个职业给出概率，并分条描述目前市场状况、匹配度和未来发展：{resume}"

        try:
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"