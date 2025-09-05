# QUICKMATE
**Care Without Delay**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)

---

## 📖 Overview
QUICKMATE is a healthcare management and monitoring system designed to ensure **transparency, accountability, and patient safety** in hospitals.  
It addresses critical problems such as **medication errors, lack of visibility in patient care, unfair billing, inaccessible consultations, emergency delays, rare blood group availability, and illegible prescriptions.**  

By combining **Flask, SQLite, AI tools, and real-time communication**, QUICKMATE bridges the gap between hospitals, doctors, patients, and their families.

---

## 🚑 Problem Statements & Solutions

### 1. Medication Errors
**Problem:** Patients risk receiving the wrong medicine, wrong dosage, or treatment errors.  
**Solution:** Nurses upload medicine photos + text details, doctors upload prescriptions, and families can **verify updates live**. Records are digitally stored with patient details for **trust & accountability**.

### 2. Lack of Visibility in Patient Care
**Problem:** Families, especially ICU patient relatives, cannot see real-time care.  
**Solution:** Secure **bedside monitoring** (camera/smartphone with encrypted streaming) allows verified families to view live updates while ensuring **privacy & consent**.

### 3. Unfair Billing Practices
**Problem:** Hospitals inflate charges and block discharge until bills are cleared.  
**Solution:** Every medicine/service is digitally recorded in **real-time**, with costs verified at each stage and visible to families, ensuring **fair billing**.

---

### 4. Costly & Inaccessible Consultations
**Problem:** Private consultations are expensive, government ones slow.  
**Solution:** QUICKMATE connects patients with **verified MBBS students/trainees (“Junior Doctors”)** for ₹30–₹50, making care **affordable & accessible**.

### 5. Delays in Emergency Care
**Problem:** Patients face delays in critical emergency wards.  
**Solution:** Families/bystanders/ambulance staff submit a **digital emergency form** that alerts hospital doctors instantly, ensuring **preparedness before arrival**.

### 6. Difficulty in Finding Rare Blood Groups
**Problem:** Outdated donor lists waste time in critical cases.  
**Solution:** QUICKMATE provides a **real-time donor list**, auto-hiding unavailable donors for 3 months per medical guidelines.

---

### 7. Illegible Prescriptions & Report Delays
**Problem:** Handwritten prescriptions and slow report processing delay care.  
**Solution:** QUICKMATE converts **handwritten prescriptions to digital text**, enables **scanned uploads of X-rays, MRIs, ultrasounds**, and provides a **first-aid scanner** for instant guided care.

---

## ✨ Key Features
- 📸 **Medicine & prescription verification** with photo & text  
- 🎥 **Live ICU monitoring** (secure, encrypted streaming)  
- 💳 **Transparent billing system** with digital proof  
- 👨‍⚕️ **Low-cost consultations** with verified junior doctors  
- 🚨 **Emergency alerts** for hospitals before patient arrival  
- 🩸 **Real-time rare blood donor registry**  
- 📝 **Prescription digitization & medical report uploads**  
- 🩹 **AI-powered first aid guidance**

---

QUICKMATE/
│── static/
│ ├── uploads/
│ ├── doctors.png
│ ├── logo.png
│
│── templates/
│ ├── admin_dashboard.html
│ ├── admin_login.html
│ ├── blood_donation.html
│ ├── charges.html
│ ├── consult.html
│ ├── emergency.html
│ ├── home.html
│ ├── icu_view.html
│ ├── index.html
│ ├── login_option.html
│ ├── login.html
│ ├── medicine_record.html
│ ├── signup.html
│
│── app.py
│── users.db
│── uploaded_image.jpg
│── requirements.txt
│── README.md


---

## ⚙️ Installation & Setup


### 1️⃣ Clone the repository
git clone https://github.com/Vishma333/QUICKMATE.git
cd QUICKMATE

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py

App will start at 👉 http://127.0.0.1:5000/
🛠 Technical Details

Backend: Flask (Python)

Database: SQLite (users.db) with SQLAlchemy ORM

Frontend: HTML, CSS, JavaScript (Jinja2 templates)

Image Processing: Pillow

Authentication: Flask-Login, Bcrypt

APIs (Future Integration): Gemini API, Ollama Mistral 7B, Agno Agent for AI report scanning

Deployment: Akash Networks

📦 Requirements
flask==3.0.3
flask_sqlalchemy==3.1.1
flask_wtf==1.2.1
flask_login==0.6.3
flask_bcrypt==1.0.1
wtforms==3.1.2
sqlalchemy==2.0.34
pillow==10.4.0
requests==2.32.3
gunicorn==23.0.0
python-dotenv==1.0.1

Install with:
pip install -r requirements.txt

🚀 Usage Guide

Admin Login → Manage hospitals, doctors, and patients

Patient Signup/Login → Access prescriptions, billing, consultations

Doctor Portal → Upload prescriptions, reports, consultation notes

Blood Donation Portal → Register as donor & search for rare blood groups

Emergency Tab → Notify hospitals with patient condition instantly

ICU View → Live secure monitoring for patient relatives

🤝 Contribution

We welcome contributions to improve QUICKMATE.

Fork the repository

Create a feature branch (git checkout -b feature-name)

Commit changes (git commit -m "Added feature XYZ")

Push to branch (git push origin feature-name)

Open a Pull Request

📜 License

This project is licensed under the MIT License – feel free to use, modify, and distribute with proper attribution.

🏥 QUICKMATE – Care Without Delay

Building trust, transparency, and safety in healthcare.

## 🗂 Project Structure
