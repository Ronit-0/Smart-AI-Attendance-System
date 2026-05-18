# 🚀 Smart AI Attendance & Mask Detection System

An enterprise-grade, fully automated computer vision kiosk that tracks faces, verifies identities using deep learning, checks for safety mask compliance, and logs attendance records in real-time. Built with a modern glassmorphism UI.

## ✨ Key Features
* **Touchless Kiosk Mode:** Endless scanning loop with a 2-second auto-focus timer and a 3-second cooldown between users. Zero human interaction required.
* **Real-Time Mask Detection:** A custom-trained MobileNetV2 architecture classifies mask compliance before logging attendance.
* **Biometric Identity Validation:** Uses DeepFace (Facenet) to extract 128-dimensional facial encodings and compare them against a secure local database.
* **Automated Data Logging:** Instantly generates daily CSV logs tracking names, exact timestamps, and mask compliance status.
* **Encrypted Admin Dashboard:** Secure login portal for database management, log viewing, and high-resolution biometric user registration.

## 🛠️ Technology Stack
* **Core Logic:** Python
* **UI/UX Dashboard:** Streamlit
* **Face Detection:** Google MediaPipe (BlazeFace)
* **Identity Recognition:** DeepFace (Facenet)
* **Mask Classification:** TensorFlow / Keras (MobileNetV2)
* **Image Processing:** OpenCV & NumPy
* **Data Management:** Pandas

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/Smart-AI-Attendance-System.git](https://github.com/YOUR_USERNAME/Smart-AI-Attendance-System.git)
cd Smart-AI-Attendance-System
```

**2. Install dependencies:**
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

**3. Run the application:**
```bash
streamlit run app.py
```

## 📖 User Guide
1. Launch the app and navigate to the **Admin Login** tab. 
2. Enter the default master key: `admin123`.
3. Go to the **Registration** tab to scan and add new users to the database (Make sure they are not wearing a mask during registration).
4. Return to the **Live Attendance** tab and click "Initialize Camera Scanner" to start the automated kiosk.

## 👥 Credits & Acknowledgments
This project was developed as an academic project during the Artificial Intelligence Programming Assistant (AIPA) course. 

* **Developers:** Ronit Das, Puskar Mandal, Poulymi Samanta, Jayanti Jana
* **Mentor:** Sayanti Manna
* **Institution:** NSTI Howrah
