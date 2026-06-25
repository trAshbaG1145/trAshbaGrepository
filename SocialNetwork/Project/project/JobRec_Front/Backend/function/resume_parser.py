from openai import OpenAI


class CareerCounselor:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def answer_question(self, question):
        try:
            # 开启流式响应
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": f"你是一位资深求职辅导员，请全面、有深度地回答以下问题：{question}"}],
                stream=True  # 开启流式响应
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"


