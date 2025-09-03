import os
import re
import random
import sqlite3
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify, Response
from werkzeug.utils import secure_filename
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from agno.agent import Agent
from agno.models.google import Gemini as AgnoGemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from fpdf import FPDF
import ollama
import cv2   # added for ICU camera

# ---------------- CONFIG ----------------
GEMINI_API_KEY = "AIzaSyCWUfG3stwj5SkAUKrVqnJoTgcVGjlzycM"
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

app = Flask(__name__)
app.secret_key = "my_flask_secret_123"

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_NAME = 'users.db'


# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    first_name TEXT,
                    middle_name TEXT,
                    last_name TEXT,
                    age INTEGER,
                    gender TEXT
                )''')
 
    # Ensure admin users exist in the database (optional but good practice)
    c.execute("SELECT * FROM users WHERE email=?", ('vishmapasayat003@gmail.com',))
    if not c.fetchone():
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", ('vishmapasayat003@gmail.com', 'Vishma@0101'))
    
    c.execute("SELECT * FROM users WHERE email=?", ('kalyanijha20.02.2008@gmail.com',))
    if not c.fetchone():
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", ('kalyanijha20.02.2008@gmail.com', 'Kalyani@2008'))
    
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# ---------------- SAFETY ----------------
SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE}
]

medical_agent = Agent(model=AgnoGemini(id="gemini-2.0-flash-exp"), tools=[DuckDuckGoTools()], markdown=True)

# ---------------- GEMINI HELPERS and PDF CREATOR... (Rest of the file is unchanged)
# --- NOTE: All helper functions like generate_text_from_image, create_pdf etc. are identical and omitted for brevity ---
def generate_text_from_image(image_file):
    with Image.open(image_file) as img:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(["Describe this injury in detail.", img], safety_settings=SAFETY_SETTINGS)
        return response.text

def generate_instructions_from_text(description):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
    You are a professional first-aid assistant. Based on this injury description:
    {description}

    Provide clear and simple step-by-step treatment instructions in this format:

    Step 1: ...
    Step 2: ...
    Step 3: ...
    Step 4: ...

    Make it practical, safe, and easy for anyone to follow. Avoid medical jargon.
    """
    response = model.generate_content([prompt], safety_settings=SAFETY_SETTINGS)
    return response.text

def instructions_to_link(instructions):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content([f"Provide a reliable link for:\n{instructions}"], safety_settings=SAFETY_SETTINGS)
    return response.text

def analyze_medical_image(image_path):
    query = """
    You are an expert in medical image analysis. Carefully examine the uploaded image and provide a detailed visual analysis. Please follow this structured format strictly:

    1. Image Type & Region
    - Describe the imaging modality, anatomy, and quality.

    2. Key Visual Findings
    - List systematic visual features and abnormalities (without diagnosis).

    3. General Visual Assessment
    - Summarize visual content without diagnosing.

    4. Patient-Friendly Explanation
    - Use non-technical terms to explain.

    5. Research Context
    - Use DuckDuckGo to find 2-3 references on similar images.

    Do not provide medical diagnoses.
    """
    agno_image = AgnoImage(filepath=image_path)
    try:
        response = medical_agent.run(query, images=[agno_image])
        content = response.content.strip()
        return content, ""
    except Exception as e:
        return f"⚠️ Analysis error: {e}", ""

def _wrap_long_tokens(text, max_len=60):
    out = []
    for token in (text or "").split():
        if len(token) <= max_len:
            out.append(token)
        else:
            chunks = [token[i:i+max_len] for i in range(0, len(token), max_len)]
            out.append(" ".join(chunks))
    return " ".join(out)

def create_pdf(patient_name, report_text):
    CLINIC_NAME = "QUICKMATE"
    CLINIC_EMAIL = "contact@quickmate.com"
    CLINIC_PHONE = "+91 6372729316, +91 6371646251"
    CLINIC_ADDRESS = "BPUT CAMPUS, CHHEND, Rourkela, Odisha - 769015"
    LOGO_PATH = os.path.join("static", "logo.png")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_fill_color(240, 248, 255)
    pdf.rect(0, 0, 210, 30, 'F')
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, 10, 6, 20)
    pdf.set_xy(35, 8)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 8, CLINIC_NAME, ln=True, align='L')
    pdf.set_x(35)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, f"Email: {CLINIC_EMAIL}   |   Phone: {CLINIC_PHONE}", ln=True, align='L')
    pdf.set_x(35)
    pdf.cell(0, 6, f"Address: {CLINIC_ADDRESS}", ln=True, align='L')
    pdf.set_draw_color(30, 136, 229)
    pdf.set_line_width(1.2)
    pdf.line(10, 28, 200, 28)
    pdf.set_draw_color(255, 111, 0)
    pdf.set_line_width(0.8)
    pdf.line(10, 31, 200, 31)
    pdf.ln(12)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, "Medical Image Analysis Report", ln=True, align='C')
    pdf.ln(2)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_id = f"MSA-{random.randint(100000, 999999)}"
    def info_row(label, value, w_label=60, w_value=115):
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(w_label, 8, f"{label}", border=1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(w_value, 8, f"{value}", border=1, ln=True)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(60, 8, "Field", border=1, fill=True, align='C')
    pdf.cell(115, 8, "Value", border=1, fill=True, align='C', ln=True)
    info_row("Patient Name", patient_name)
    info_row("Report ID", report_id)
    info_row("Generated On", current_time)
    pdf.ln(4)
    def parse_sections(text):
        labels = ["Injury:", "Instructions:", "Reference:", "Analysis:"]
        sections = {}
        text = text or ""
        for i, label in enumerate(labels):
            start = text.find(label)
            if start != -1:
                end = len(text)
                for j in range(i + 1, len(labels)):
                    nxt = text.find(labels[j])
                    if nxt != -1 and nxt > start:
                        end = min(end, nxt)
                content = text[start + len(label):end].strip()
                if content:
                    sections[label[:-1]] = content
        return sections
    sections = parse_sections(report_text)
    for title in ["Injury", "Instructions", "Reference", "Analysis"]:
        if title in sections and sections[title].strip():
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(30, 136, 229)
            pdf.cell(0, 8, title, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.4)
            y = pdf.get_y()
            pdf.line(10, y, 200, y)
            pdf.ln(2)
            pdf.set_font('Arial', '', 11)
            content = _wrap_long_tokens(sections[title], max_len=60)
            for block in content.split('\n'):
                block = block.strip()
                if not block:
                    continue
                pdf.multi_cell(0, 7, block)
            pdf.ln(2)
    pdf.set_y(-30)
    pdf.set_draw_color(30, 136, 229)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, "Note: This is an AI-assisted report for informational purposes. Please consult a qualified medical professional for diagnosis and treatment.")
    pdf.set_text_color(0, 0, 0)
    pdf_output = os.path.join(app.config['UPLOAD_FOLDER'], f"{patient_name.replace(' ', '_')}_report.pdf")
    pdf.output(pdf_output)
    return pdf_output

# ---------------- ICU CAMERA STREAM ----------------
PHONE_STREAM_URL = os.environ.get('PHONE_STREAM_URL', 'http://192.168.40.183:8080/video')
icu_cap = None
icu_cap_lock = threading.Lock()
def open_icu_capture():
    global icu_cap
    with icu_cap_lock:
        if icu_cap is None or not icu_cap.isOpened():
            icu_cap = cv2.VideoCapture(PHONE_STREAM_URL)
def close_icu_capture():
    global icu_cap
    with icu_cap_lock:
        if icu_cap is not None:
            try:
                icu_cap.release()
            except Exception:
                pass
            icu_cap = None
def read_icu_frame():
    with icu_cap_lock:
        if icu_cap is None:
            return False, None
        return icu_cap.read()
def generate_icu_mjpeg():
    open_icu_capture()
    time.sleep(0.25)
    while True:
        success, frame = read_icu_frame()
        if not success or frame is None:
            close_icu_capture(); time.sleep(0.5); open_icu_capture(); continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

@app.route('/icu_feed')
def icu_feed():
    if not session.get('logged_in'):
        return Response("Unauthorized", status=403)
    return Response(generate_icu_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/icu_snapshot', methods=['POST'])
def icu_snapshot():
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "Not logged in"}), 403
    open_icu_capture()
    success, frame = read_icu_frame()
    if not success or frame is None:
        return jsonify({"ok": False, "error": "No frame available"}), 500
    ts = int(time.time())
    filename = f"icu_snapshot_{ts}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)
    return jsonify({"ok": True, "file": f"/{filepath.replace(os.sep, '/')}"}), 200

# ---------------- AUTH ROUTES ----------------
@app.route('/')
def home():
    return redirect(url_for('login_option'))

@app.route('/login_option')
def login_option():
    return render_template('login_option.html')

# MODIFIED admin_login ROUTE
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    # This block handles the form submission
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        admin_credentials = {
            'vishmapasayat003@gmail.com': 'Vishma@0101',
            'kalyanijha20.02.2008@gmail.com': 'Kalyani@2008'
        }

        if email in admin_credentials and password == admin_credentials[email]:
            session['logged_in'] = True
            session['is_admin'] = True
            session['user_email'] = email
            return redirect(url_for('admin_dashboard')) # Success -> dashboard
        else:
            return render_template('admin_login.html', error='Invalid admin credentials')
    
    # If it's a GET request (just clicking the link), always show the login page.
    # The problematic redirect logic has been removed from here.
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('logged_in') or not session.get('is_admin'):
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
            user = c.fetchone()
        if user:
            session['logged_in'] = True
            session['is_admin'] = False 
            session['user_email'] = email
            return redirect(url_for('clinic'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        fname = request.form.get('first_name')
        mname = request.form.get('middle_name')
        lname = request.form.get('last_name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        try:
            with sqlite3.connect(DB_NAME, timeout=10) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (email, password, first_name, middle_name, last_name, age, gender) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (email, password, fname, mname, lname, age, gender))
                conn.commit()
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Email already exists")
    return render_template('signup.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login_option'))

# ---------------- PROTECTED ROUTES ----------------
@app.route('/clinic')
def clinic():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    if session.get('is_admin'):
        return redirect(url_for('admin_dashboard'))
    return render_template("home.html")

# --- All other routes (emergency, blood_donation, etc.) are unchanged ---
@app.route('/emergency')
def emergency():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    return render_template('emergency.html')

@app.route('/blood_donation')
def blood_donation():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    return render_template('blood_donation.html')

@app.route('/consult')
def consult():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    return render_template('consult.html')

@app.route('/medicine_record')
def medicine_record():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    return render_template('medicine_record.html') 

@app.route('/icu_view')
def icu_view():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    return render_template('icu_view.html')

@app.route('/charges')
def charges():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    return render_template('charges.html')

@app.route('/scanning', methods=['GET', 'POST'])
def scanning():
    if not session.get('logged_in'):
        return redirect(url_for('login_option'))
    if request.method == 'POST':
        action_type = request.form.get('action_type')
        patient_name = request.form.get('patient_name', 'Unknown')
        file = request.files.get('image_file')
        if not file or not allowed_file(file.filename):
            return render_template("index.html", result_text="⚠️ Please upload a valid JPG/PNG file.")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        injury_text, instructions_text, link_text, result_text, pdf_link = None, None, None, None, None
        try:
            if action_type == "injury_analysis":
                injury_text = generate_text_from_image(filepath)
                instructions_text = generate_instructions_from_text(injury_text)
                link_text = instructions_to_link(instructions_text)
            elif action_type == "medical_report":
                result_text = analyze_medical_image(filepath)[0]
            else:
                result_text = "⚠️ Unknown action type selected."
            report_content = "\n".join(filter(None, [
                f"Injury: {injury_text}" if injury_text else "",
                f"Instructions:\n{instructions_text}" if instructions_text else "",
                f"Reference: {link_text}" if link_text else "",
                f"Analysis:\n{result_text}" if result_text else "",
            ]))
            pdf_path = create_pdf(patient_name, report_content)
            pdf_link = f"/{pdf_path.replace(os.sep, '/')}"
        except Exception as e:
            result_text = f"⚠️ Error during scan: {str(e)}"
        return render_template("index.html",
                               injury_text=injury_text,
                               instructions_text=instructions_text,
                               link_text=link_text,
                               result_text=result_text,
                               pdf_link=pdf_link)
    return render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if not session.get('logged_in'):
        return jsonify({"error": "Not logged in"}), 403
    user_message = request.form.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({'role': 'user', 'content': user_message})
    session['chat_history'] = session['chat_history'][-10:]
    try:
        response = ollama.chat(model='mistral', messages=session['chat_history'])
        reply = response.get('message', {}).get('content', "").strip()
        if reply:
            session['chat_history'].append({'role': 'assistant', 'content': reply})
        else:
            reply = "⚠️ I could not generate a response. Please try again."
        session.modified = True
        return jsonify({"reply": reply})
    except Exception as e:
        app.logger.error(f"Chatbot error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    app.run(debug=True)