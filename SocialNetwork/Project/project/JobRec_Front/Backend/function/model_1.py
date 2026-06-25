from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 加载预训练模型和分词器
model_name_promptclue = "ClueAI/PromptCLUE-base"
tokenizer_promptclue = AutoTokenizer.from_pretrained(model_name_promptclue)
model_promptclue = AutoModelForSequenceClassification.from_pretrained(model_name_promptclue)

model_name_roberta = "hfl/chinese-roberta-wwm-ext"
tokenizer_roberta = AutoTokenizer.from_pretrained(model_name_roberta)
model_roberta = AutoModelForSequenceClassification.from_pretrained(model_name_roberta)

model_name_macbert = "hfl/chinese-macbert-base"
tokenizer_macbert = AutoTokenizer.from_pretrained(model_name_macbert)
model_macbert = AutoModelForSequenceClassification.from_pretrained(model_name_macbert)

# 创建文本分类管道
classifier_promptclue = pipeline("text-classification", model=model_promptclue, tokenizer=tokenizer_promptclue)
classifier_roberta = pipeline("text-classification", model=model_roberta, tokenizer=tokenizer_roberta)
classifier_macbert = pipeline("text-classification", model=model_macbert, tokenizer=tokenizer_macbert)

# 定义关键点提取函数
def extract_key_points(resume_text):
    # 使用PromptCLUE-base模型进行关键点提取
    result_promptclue = classifier_promptclue(resume_text)
    
    # 使用chinese-roberta-wwm-ext模型进行关键点提取
    result_roberta = classifier_roberta(resume_text)
    
    # 使用chinese-macbert-base模型进行关键点提取
    result_macbert = classifier_macbert(resume_text)
    
    # 整合结果
    key_points = {
        "PromptCLUE-base": result_promptclue,
        "chinese-roberta-wwm-ext": result_roberta,
        "chinese-macbert-base": result_macbert
    }
    
    return key_points

# 示例使用
resume_text = "张三，男，30岁，毕业于北京大学计算机系，拥有10年软件开发经验，熟练掌握Python、Java等编程语言，曾就职于腾讯、阿里巴巴等知名公司，负责多个大型项目的开发与管理。"
key_points = extract_key_points(resume_text)
print(key_points)
