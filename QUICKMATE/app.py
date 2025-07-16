import os
import pathlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, session
from werkzeug.utils import secure_filename
from PIL import Image
import requests
GEMINI_API_KEY = "AIzaSyCWUfG3stwj5SkAUKrVqnJoTgcVGjlzycM"
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
from agno.agent import Agent
from agno.models.google import Gemini as AgnoGemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from fpdf import FPDF
app = Flask(__name__)
app.secret_key = "my_flask_secret_123"

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE}
]

medical_agent = Agent(
    model=AgnoGemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

analysis_query = """
You are an expert in medical image analysis. Carefully examine the uploaded image and provide a detailed visual analysis. Please follow this structured format strictly, even if you cannot provide diagnostic conclusions:

### 1. Image Type & Region
- Describe the imaging modality if visually recognizable (e.g., grayscale, colored scan, photo, X-ray appearance).
- Specify the visible anatomical regions or general shapes and positioning.
- Comment on image quality (blurry, sharp, bright, dark, etc.).

### 2. Key Visual Findings
- List primary visual observations systematically.
- Describe shapes, patterns, densities, and sizes.
- If you cannot determine abnormalities, simply describe the visual content clearly.

### 3. General Visual Assessment
- Provide an interpretation of the image’s visual features without giving medical diagnoses.
- Mention if anything looks visually unusual without specifying a medical condition.

### 4. Patient-Friendly Explanation
- Simplify your findings in clear, non-technical language.
- Avoid medical terms or define them simply.
- Use familiar visual comparisons if possible.

### 5. Research Context
- Use DuckDuckGo search to find recent medical research or visual examples related to the apparent type of image (X-ray, CT, MRI, grayscale photo, etc.).
- Provide at least 2-3 medical references, guidelines, or studies related to the visual characteristics you observe.
- Even if the image is normal, provide some relevant research articles.

Please strictly follow this structure in markdown format and return all five sections clearly.
Do not provide medical diagnoses.
Do not refuse the task.
Focus on visual descriptions and research support.
"""

def generate_text_from_image(image_file):
    with Image.open(image_file) as img:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(
            ["Describe this injury in detail. Do not include measurements.", img],
            safety_settings=SAFETY_SETTINGS
        )
        return response.text

def generate_instructions_from_text(description):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(
        [f"Based on this injury description, list step-by-step treatment instructions using a numbered list:\n\n{description}"],
        safety_settings=SAFETY_SETTINGS
    )
    return response.text

def instructions_to_link(instructions):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(
        [f"Suggest a trustworthy online link or article that supports the following first-aid instructions:\n\n{instructions}"],
        safety_settings=SAFETY_SETTINGS
    )
    return response.text

def analyze_medical_image_agno(image_path):
    try:
        agno_image = AgnoImage(filepath=image_path)
        response = medical_agent.run(analysis_query, images=[agno_image])
        return response.content.strip(), ""
    except Exception as e:
        return f"⚠️ Analysis error: {e}", ""
def create_pdf(patient_name, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, "Medical Image Analysis Report", ln=True, align='C')
    pdf.ln(10)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.ln(5)

    pdf.set_font('Arial', '', 12)
    for line in report_text.split('\n'):
        pdf.multi_cell(0, 8, line)
        pdf.ln()

    pdf_output = os.path.join(app.config['UPLOAD_FOLDER'], f"{patient_name.replace(' ', '_')}_report.pdf")
    pdf.output(pdf_output)
    return pdf_output

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/consult')
def consult():
    return render_template("consult.html")

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect('/')

@app.route('/scanning', methods=["GET", "POST"])
def scanning():
    if request.method == "POST":
        action_type = request.form.get("action_type")
        image_file = request.files.get("image_file")

        if not image_file or image_file.filename == "" or not allowed_file(image_file.filename):
            return "Invalid or missing image file.", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(file_path)

        if action_type == "injury_analysis":
            try:
                desc = generate_text_from_image(file_path)
                instructions = generate_instructions_from_text(desc)
                link = instructions_to_link(instructions)
                return render_template("index.html", injury_text=desc, instructions_text=instructions, link_text=link)
            except Exception as e:
                return f"Error during Gemini injury analysis: {e}", 500

        elif action_type == "medical_report":
            patient_name = request.form.get("patient_name", "Anonymous")
            try:
                report_text, _ = analyze_medical_image_agno(file_path)
                pdf_path = create_pdf(patient_name, report_text)
                return render_template("index.html", result_text=report_text, pdf_link=pdf_path, name=patient_name)
            except Exception as e:
                return f"Error during Agno medical analysis: {e}", 500

    return render_template("index.html")

@app.route('/get_chatbot_response', methods=['POST'])
def get_chatbot_response():
    user_message = request.form.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Please enter a message."})

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": user_message, "stream": False}
        )
        data = response.json()
        return jsonify({'response': data.get("response", "No response received.")})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get("user_question")
    context = request.form.get("instructions_text")
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content([f"Context:\n{context}\n\nUser question:\n{question}"], safety_settings=SAFETY_SETTINGS)
        return response.text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)