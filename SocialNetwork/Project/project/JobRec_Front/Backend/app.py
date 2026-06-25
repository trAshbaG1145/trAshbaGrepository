import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from flask import Flask, request, jsonify, render_template, Response
from openai import OpenAI

from Backend.function.resume_parser import CareerCounselor
from Backend.function.job_rec import JobRecommender
from KG_answer import recommend_jobs
from KG_search import search_jobs
from flask_cors import CORS
from KG_processing import process_sentence
from upload import process_file, extract_text_from_pdf, extract_text_from_pdf_ocr, extract_text_from_word, extract_text_from_image
from Backend.function.career_analyzer import CareerTimelineAnalyzer
app = Flask(__name__)
CORS(app)
url = 'https://api.siliconflow.cn/v1/'
# 设置 OpenAI API 客户端
api_key = 'sk-wxigmetfuzoxfyvrgmsrrcaffxdwtapdwdjckgguhrlvkonp'


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('user_input')
    experience_input = data.get('experience_input')
    salary_input = data.get('salary_input')

    # 调用句子处理模块
    skills, education = process_sentence(user_input)

    # 获取推荐职业和 ECharts 数据
    recommended_jobs, echart_data = recommend_jobs(skills,
                                                   education=education,
                                                   experience=experience_input,
                                                   min_salary=salary_input)

    if recommended_jobs:
        result = {
            "message": "根据你提供的信息，推荐的职业及其相关信息如下：",
            "jobs": recommended_jobs,
            "echart_data": echart_data
        }
    else:
        result = {
            "message": "未找到匹配的职业。",
            "jobs": [],
            "echart_data": None
        }

    return jsonify(result)


@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    job_name = data.get('job_name')
    skills = data.get('skills', [])
    education = data.get('education')
    experience = data.get('experience')

    top_jobs, echart_data = search_jobs(job_name, skills, education, experience)
    return jsonify({
        "top_jobs": top_jobs,
        "echart_data": echart_data
    })


app.config['UPLOAD_FOLDER'] = 'uploads'  # 上传文件的保存目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

job_recommender = JobRecommender(base_url=url, api_key=api_key)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    处理前端上传的文件
    """
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if file:
        # 保存上传的文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 提取文本
        extracted_text = process_file(file_path)

        # 保存提取的文本到txt文件
        output_file = os.path.splitext(file_path)[0] + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        result = job_recommender.jobRecommender(extracted_text)

        return jsonify({
            'message': '文件处理成功',
            'text': extracted_text,
            'output_file': output_file,
            'result': result

        })


# 初始化简历解析器
resume_parser = CareerCounselor(base_url=url, api_key=api_key)


@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required"}), 400

    def generate():
        yield from resume_parser.answer_question(question)

    return Response(generate(), mimetype='text/event-stream')


analyzer = CareerTimelineAnalyzer(base_url=url, api_key=api_key)


@app.route('/api/analyze', methods=['POST'])
def analyze_career():
    data = request.get_json()
    career_name = data.get('career_name')

    if not career_name:
        return jsonify({"error": "职业名不能为空"}), 400

    timeline = analyzer.analyze_career_timeline(career_name)

    return jsonify({"timeline": timeline})


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("FLASK_PORT", "8081")))
