from app import app
from flask import request, jsonify
from models.similarity_calc import calculate_similarity
from models.cv_classify import classify_resume
import PyPDF2
import requests
from io import BytesIO


@app.route("/cv-classify", methods=['post'])
def calculate_job_matching():
    if request.method.__eq__("POST"):
        url = request.form.get("cv_link")
        response = requests.get(url)

        pdf_file = BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)

        pdf_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text() + "\n"

        return classify_resume(pdf_text)


@app.route("/cv-matching-jd", methods=['post'])
def read_cv():
    if request.method.__eq__("POST"):
        url = request.form.get("cv_link")
        jd = request.form.get("jd_text")
        print("url la: ", url)
        print("jd la: ", jd)
        response = requests.get(url)

        pdf_file = BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)

        pdf_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text() + "\n"

        return str(calculate_similarity(jd, pdf_text) * 100)


@app.route("/cv-classify-file", methods=['post'])
def calculate_job_cv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    text = ""

    if file and allowed_file(file.filename):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'

        return jsonify({"cvType": classify_resume(text), "code": 200})
    else:
        return jsonify({"error": "File type not allowed"}), 400

# @app.route("/cv-matching-jd", methods=['post'])
# def read_cv():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     jd = ""
#     text = ""
#     if request.method.__eq__("POST"):
#         jd = request.form.get("jd_text")
#
#     if file and allowed_file(file.filename):
#         # Đọc nội dung PDF
#         pdf_reader = PyPDF2.PdfReader(file)
#         for page in pdf_reader.pages:
#             text += page.extract_text() + '\n'
#
#         return jsonify({"similarity": str(calculate_similarity(jd, text) * 100), "code": 200})
#     else:
#         return jsonify({"error": "File type not allowed"}), 400


def allowed_file(filename):
    allowed_extensions = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == "__main__":
    app.run(debug=True)
