from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline
from PIL import Image


def process_resume(image_path):
    try:
        # 1. OCR 提取文本
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 2. 生成摘要
        # 使用 fnlp/bart-large-chinese 模型进行摘要生成
        summarizer = pipeline("summarization", model="fnlp/bart-large-chinese", tokenizer="fnlp/bart-large-chinese")
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]

        # 3. 提取关键点
        ner_pipeline = pipeline("ner", model="ckiplab/bert-base-chinese-ner", tokenizer="ckiplab/bert-base-chinese-ner",
                                aggregation_strategy="average")
        entities = ner_pipeline(text)
        key_points = {
            "教育背景": [ent["word"] for ent in entities if ent["entity_group"] == "EDUCATION"],
            "工作经历": [ent["word"] for ent in entities if ent["entity_group"] == "WORK_EXPERIENCE"],
            "技能": [ent["word"] for ent in entities if ent["entity_group"] == "SKILL"]
        }

        return {
            "原始文本": text,
            "摘要": summary,
            "关键点": key_points
        }
    except Exception as e:
        print(f"处理简历时发生错误: {e}")
        return None


if __name__ == "__main__":
    image_path = "resume.png"
    result = process_resume(image_path)
    if result:
        print("原始文本:", result["原始文本"])
        print("摘要:", result["摘要"])
        print("关键点:", result["关键点"])
