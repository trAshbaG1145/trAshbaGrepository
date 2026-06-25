import os
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract
from pdf2image import convert_from_path


# 设置Tesseract的路径（如果需要）
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf(pdf_path):
    """
    提取PDF文件中的文本
    """
    text = ""
    try:
        # 尝试直接提取文本
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()

        # 如果直接提取的文本为空，则尝试使用OCR
        if not text.strip():
            text = extract_text_from_pdf_ocr(pdf_path)
    except Exception as e:
        print(f"直接提取PDF文本失败：{e}，将尝试使用OCR提取")
        text = extract_text_from_pdf_ocr(pdf_path)

    return text


def extract_text_from_pdf_ocr(pdf_path):
    """
    使用OCR提取PDF文件中的文本（适用于扫描件）
    """
    text = ""
    try:
        # 将PDF转换为图片
        images = convert_from_path(pdf_path, dpi=300)
        for image in images:
            # 使用Tesseract OCR识别图片中的文本
            text += pytesseract.image_to_string(image, lang='chi_sim+eng')  # 根据需要选择语言包
    except Exception as e:
        print(f"使用OCR提取PDF文本失败：{e}")

    return text


def extract_text_from_word(doc_path):
    """
    提取Word文件中的文本
    """
    text = ""
    try:
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"提取Word文本失败：{e}")

    return text


def extract_text_from_image(image_path):
    """
    使用OCR从图片中提取文本
    """
    try:
        # 打开图片
        image = Image.open(image_path)

        # 使用Tesseract OCR识别图片中的文本
        text = pytesseract.image_to_string(image, config='--psm 11', lang='chi_sim+eng').replace(' ', '')

        return text
    except Exception as e:
        print(f"从图片提取文本失败：{e}")
        return ""


def process_file(file_path):
    """
    处理文件，提取文本
    """
    text = ""
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_word(file_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        text = extract_text_from_image(file_path)
    else:
        print("不支持的文件格式")

    return text
