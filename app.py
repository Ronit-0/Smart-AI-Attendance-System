import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import pandas as pd
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
import time
import base64 

# ==========================================
# 1. PAGE CONFIG, STATE & UNIFIED CSS
# ==========================================
st.set_page_config(page_title="Smart Attendance System", layout="wide", initial_sidebar_state="collapsed")

# Initialize Admin Login State
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False

st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

/* --- PROFESSIONAL GRADIENT BACKGROUND --- */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
    background-attachment: fixed !important;
    color: #f8fafc;
}

.block-container { max-width: 1200px !important; padding-top: 2rem !important; }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(5, 8, 15, 0.9); }
::-webkit-scrollbar-thumb { background: rgba(0, 255, 255, 0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 255, 0.8); }

/* --- HEADING COLORS (Uniform Cyan Glow) --- */
.main-title { 
    font-size: 3.5rem; 
    font-weight: 900; 
    text-align: center; 
    color: #00FFFF; 
    text-shadow: 0 0 15px rgba(0, 255, 255, 0.5); 
    letter-spacing: 2px; 
    margin-bottom: 0px;
}
.sub-title { text-align: center; color: #94a3b8; font-size: 1.2rem; margin-top: 5px; margin-bottom: 30px; font-weight: 300; }

/* --- INPUT BOX GLOW --- */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
    border: 1px solid rgba(0, 255, 255, 0.8) !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.4) !important;
}
input { color: #ffffff !important; }

/* --- CENTERED NAVIGATION PILLS --- */
[data-testid="stRadio"] { 
    display: flex !important; 
    flex-direction: column !important;
    align-items: center !important; 
    justify-content: center !important;
    margin-bottom: 40px !important; 
    width: 100% !important;
}
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] label > div:first-child { display: none !important; } 

div[role="radiogroup"] {
    display: flex !important; 
    justify-content: center !important;
    align-items: center !important;
    margin: 0 auto !important;
    gap: 10px !important; 
    padding: 10px !important;
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 50px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5), inset 0 0 0.5px rgba(255,255,255,0.1) !important;
}
div[role="radiogroup"] label {
    padding: 12px 25px !important; border-radius: 40px !important;
    color: #94A3B8 !important; font-weight: 600 !important; font-size: 1rem !important;
    background: transparent !important; transition: all 0.3s ease !important; cursor: pointer !important;
}
div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.08) !important; color: #ffffff !important; transform: translateY(-2px);
}
div[role="radiogroup"] label:has(input:checked) {
    background: rgba(0, 255, 255, 0.15) !important; color: #00FFFF !important;
    border: 1px solid rgba(0, 255, 255, 0.4) !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3), inset 0 0 10px rgba(0, 255, 255, 0.1);
}

/* --- GLASS CARDS FOR DETAILED VIEWS --- */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(0, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
    transition: transform 0.3s ease, border 0.3s ease;
    height: 100%;
}
.glass-card:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(0, 255, 255, 0.4);
}
.glass-card h4 { color: #00FFFF; margin-bottom: 15px; font-weight: 700; }
.glass-card p { color: #cbd5e1; line-height: 1.6; }

@keyframes borderMovingGlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}
.glow-box { 
    background: linear-gradient(145deg, rgba(20,25,35,0.8) 0%, rgba(10,15,25,0.95) 100%); 
    backdrop-filter: blur(15px); border-radius: 16px; padding: 25px; 
    position: relative; overflow: hidden; margin-bottom: 20px;
}
.glow-box::after {
    content: ''; position: absolute; inset: 0; border-radius: 16px; padding: 2px; 
    background: linear-gradient(90deg, #00FFFF, transparent, #3B82F6, transparent, #00FFFF);
    background-size: 300% 100%;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: destination-out; mask-composite: exclude;
    animation: borderMovingGlow 6s linear infinite; pointer-events: none; opacity: 0.6;
}

img { border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETUP & MODEL LOADING (OPTIMIZED)
# ==========================================
DB_PATH = "registered_users"
for folder in [DB_PATH, "attendance_logs"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

model_path = 'blaze_face_short_range.tflite'
if not os.path.exists(model_path):
    st.info("Downloading MediaPipe Face Detection model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", model_path)

import zipfile
import os

@st.cache_resource
def load_ai_models():

    # Extract model if not already extracted
    if not os.path.exists("robust_mask_detector.h5"):
        with zipfile.ZipFile("robust_mask_detector.zip", "r") as zip_ref:
            zip_ref.extractall(".")

    try:
        mask_model = load_model("robust_mask_detector.h5", compile=False)

    except Exception as e:
        mask_model = None
        st.error(f"Actual model loading error: {e}")

    return mask_model
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
    face_detector = vision.FaceDetector.create_from_options(options)
    
    # --- PERFORMANCE UPGRADE ---
    # Pre-loading the Facenet model into cache here prevents the massive lag spike 
    # that usually happens on the first camera scan!
    try:
        DeepFace.build_model("Facenet")
    except:
        pass
    
    return mask_model, face_detector

mask_model, face_detector = load_ai_models()

# ==========================================
# 3. HEADER & DYNAMIC ROUTING
# ==========================================
st.markdown('<div class="main-title">SMART ATTENDANCE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by DeepFace & MobileNetV2</div>', unsafe_allow_html=True)

# Navigation Menu Changes Based on Login Status
if not st.session_state["admin_logged_in"]:
    choice = st.radio("Navigation", ["🏠 Home", "🔴 Live Attendance", "🧠 Model Overview", "📄 Documentation", "🔐 Admin Login"], horizontal=True, label_visibility="collapsed")
else:
    choice = st.radio("Navigation", ["👤 Registration", "📊 Log Records", "👥 Registered Users", "🚪 Logout"], horizontal=True, label_visibility="collapsed")

# ==========================================
# 4. PUBLIC VIEWS
# ==========================================
if choice == "🏠 Home":
    st.markdown("""
    <div class="glow-box">
        <h3 style='color:#00FFFF; text-align:center; margin-bottom:30px;'>Welcome to the Next Generation of Access Control</h3>
        <div style="display:flex; justify-content:space-around; text-align:center;">
            <div><h1 style="color:#ffffff;">🚀</h1><p style="color:#cbd5e1;">Fully Automated Kiosk</p></div>
            <div><h1 style="color:#ffffff;">🛡️</h1><p style="color:#cbd5e1;">Mask Compliance Checking</p></div>
            <div><h1 style="color:#ffffff;">⚡</h1><p style="color:#cbd5e1;">Sub-second Identification</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>For Users</h4>
            <p>Simply navigate to the <b>Live Attendance</b> tab and step into the frame. The system will automatically detect your face, verify your identity against the database, check your mask compliance, and log your attendance. No clicking required.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>For Administrators</h4>
            <p>Select the <b>Admin Login</b> tab to authenticate. Once logged in, you will have exclusive access to register new users with high-resolution reference photos, view daily attendance logs in CSV format, and manage the active database.</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🧠 Model Overview":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>1. Face Detection</h4>
            <p><b>Model:</b> Google MediaPipe (BlazeFace)</p>
            <p>A highly optimized, sub-millisecond neural network designed specifically for mobile and edge devices. It scans the incoming video feed to extract precise facial bounding boxes before any heavy processing occurs.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>2. Mask Classification</h4>
            <p><b>Model:</b> Custom MobileNetV2</p>
            <p>A lightweight convolutional neural network trained on over 12,000 images. The detected facial region is isolated, preprocessed into RGB arrays, and passed through the network to determine safety compliance.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h4>3. Identity Validation</h4>
            <p><b>Model:</b> DeepFace (Facenet Model)</p>
            <p>The system generates a 128-dimensional mathematical encoding of the user's facial features and compares the Euclidean distance against all registered references to confidently verify identity.</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "📄 Documentation":
    st.markdown('<div class="glow-box"><h3 style="color:#00FFFF; margin-bottom:0px;">System Guidelines & Troubleshooting</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>Registration Best Practices</h4>
            <p>• <b>No Masks:</b> Ensure the subject is not wearing a mask during initial registration.</p>
            <p>• <b>Lighting:</b> Register users in a well-lit environment to capture clear facial encodings.</p>
            <p>• <b>Alignment:</b> The system requires 2 seconds (60 frames) of perfect alignment in the green target box to auto-capture.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>Live Attendance Behavior</h4>
            <p>• <b>Kiosk Mode:</b> The scanner runs endlessly. It requires zero human interaction.</p>
            <p>• <b>Cooldown Timer:</b> After a successful scan, the system initiates a 3-second cooldown (gray box) before accepting the next person.</p>
            <p>• <b>Data Logging:</b> All events are instantly saved to the <code>attendance_logs</code> folder grouped by date.</p>
        </div>
        """, unsafe_allow_html=True)

elif choice == "🔴 Live Attendance":
    registered = [f for f in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, f))]
    if len(registered) == 0:
        st.error("⚠️ No users registered in Database! Go to Admin Panel.")
    else:
        st.markdown("""
        <div class="glow-box">
            <h3 style='color:#00FFFF; margin-bottom:10px;'>Live Biometric Scan</h3>
            <p style='color:#94a3b8; margin-bottom:0px;'>📷 System is active! Step into the frame to automatically log attendance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        start_button_container = st.empty()
        
        if start_button_container.button("Initialize Camera Scanner", use_container_width=True):
            start_button_container.empty() 
            
            status_text = st.empty()
            frame_window = st.empty() 
            results_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            alignment_frames = 0
            cooldown_frames = 0 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                box_size = 250
                center_x, center_y = w // 2, h // 2
                top_left = (center_x - box_size // 2, center_y - box_size // 2)
                bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
                
                guide_color = (0, 0, 255) 
                instruction = "Waiting for subject... Position your face inside the box."
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    guide_color = (128, 128, 128) 
                    instruction = "✅ Attendance Logged! Waiting for next person..."
                    
                    cv2.rectangle(rgb_frame, top_left, bottom_right, guide_color, 3)
                    frame_window.image(rgb_frame)
                    status_text.markdown(f"**Status:** {instruction}")
                    
                    if cooldown_frames == 0:
                        results_placeholder.empty()
                    continue 
                    
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = face_detector.detect(mp_image)
                
                if detection_result.detections:
                    bbox = detection_result.detections[0].bounding_box
                    face_cx = int(bbox.origin_x + (bbox.width / 2))
                    face_cy = int(bbox.origin_y + (bbox.height / 2))
                    
                    if (top_left[0] < face_cx < bottom_right[0]) and (top_left[1] < face_cy < bottom_right[1]):
                        guide_color = (0, 255, 0) 
                        alignment_frames += 1
                        instruction = f"Scanning Biometrics... {alignment_frames}/60"
                        
                        if alignment_frames > 60:
                            status_text.markdown("**Status:** Processing Identity and Mask Compliance...")
                            
                            startX, startY = max(0, int(bbox.origin_x)), max(0, int(bbox.origin_y))
                            endX, endY = min(w, startX + int(bbox.width)), min(h, startY + int(bbox.height))
                            face_crop = frame[startY:endY, startX:endX]
                            
                            mask_status = "Unknown"
                            identity_name = "Unknown"
                            
                            if face_crop.size != 0:
                                if mask_model is not None:
                                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                    resized_face = cv2.resize(rgb_face, (224, 224))
                                    img_array = np.expand_dims(resized_face, axis=0)
                                    preprocessed_img = preprocess_input(img_array)
                                    prediction = mask_model.predict(preprocessed_img)[0][0]
                                    mask_status = "Mask ON" if prediction < 0.5 else "No Mask" 
                                
                                try:
                                    cv2.imwrite("temp_scan.jpg", face_crop)
                                    dfs = DeepFace.find(img_path="temp_scan.jpg", db_path=DB_PATH, model_name="Facenet", enforce_detection=False, silent=True)
                                    if len(dfs) > 0 and not dfs[0].empty:
                                        match_path = dfs[0].iloc[0]['identity']
                                        identity_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
                                except:
                                    pass

                            date_string = datetime.now().strftime("%Y-%m-%d")
                            time_string = datetime.now().strftime("%H:%M:%S")
                            log_file = f"attendance_logs/log_{date_string}.csv"
                            
                            log_entry = pd.DataFrame({"Name": [identity_name], "Time": [time_string], "Mask Status": [mask_status]})
                            if not os.path.isfile(log_file): log_entry.to_csv(log_file, index=False)
                            else: log_entry.to_csv(log_file, mode='a', header=False, index=False)
                            
                            with results_placeholder.container():
                                st.markdown("---")
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption="Analyzed Region")
                                with col2:
                                    if identity_name != "Unknown": st.success(f"**Identity Verified:** {identity_name}")
                                    else: st.warning("**Identity:** Unknown User")
                                    
                                    if mask_status == "Mask ON": st.success(f"**Compliance:** {mask_status} ✅")
                                    else: st.error(f"**Compliance:** {mask_status} ❌")
                            
                            alignment_frames = 0
                            cooldown_frames = 90 
                                    
                    else:
                        alignment_frames = 0
                else:
                    alignment_frames = 0
                    
                cv2.rectangle(rgb_frame, top_left, bottom_right, guide_color, 3)
                frame_window.image(rgb_frame)
                if cooldown_frames == 0: 
                    status_text.markdown(f"**Status:** {instruction}")
                
            cap.release()
            if os.path.exists("temp_scan.jpg"): os.remove("temp_scan.jpg")

elif choice == "🔐 Admin Login":
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("""
        <div class="glow-box" style="text-align:center;">
            <h2 style='color:#00FFFF; margin-bottom: 10px;'>Admin Authentication</h2>
            <p style='color:#cbd5e1; margin-bottom: 0px;'>Please enter the master key to access the database.</p>
        </div>
        """, unsafe_allow_html=True)
        
        pwd = st.text_input("Security Key", type="password", label_visibility="collapsed", placeholder="Enter Password...")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔓 Authenticate", use_container_width=True):
            if pwd == "admin123":
                st.session_state["admin_logged_in"] = True
                st.rerun()
            else:
                st.error("Access Denied. Incorrect Password.")

# ==========================================
# 5. SECURE ADMIN VIEWS (Only if Logged In)
# ==========================================
if choice == "👤 Registration":
    st.markdown("""
    <div class="glow-box">
        <h3 style='color:#00FFFF; margin-bottom:10px;'>Register New Identity</h3>
        <p style='color:#94a3b8; margin-bottom:0px;'>⚠️ Ensure no mask is worn during registration. Align face in the green box to auto-capture.</p>
    </div>
    """, unsafe_allow_html=True)
    
    new_name = st.text_input("Enter Full Name / ID (No spaces)")
    
    if new_name:
        start_reg_container = st.empty()
        if start_reg_container.button("Start Biometric Scanner", use_container_width=True):
            start_reg_container.empty()
            
            status_text = st.empty()
            frame_window = st.empty() 
            
            cap = cv2.VideoCapture(0)
            alignment_frames = 0
            captured = False
            safe_name = new_name.replace(" ", "_").strip()
            
            while cap.isOpened() and not captured:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                box_size = 250
                center_x, center_y = w // 2, h // 2
                top_left = (center_x - box_size // 2, center_y - box_size // 2)
                bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
                
                guide_color = (0, 0, 255)
                instruction = "Position your face inside the box."
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = face_detector.detect(mp_image)
                
                if detection_result.detections:
                    bbox = detection_result.detections[0].bounding_box
                    face_cx = int(bbox.origin_x + (bbox.width / 2))
                    face_cy = int(bbox.origin_y + (bbox.height / 2))
                    
                    if (top_left[0] < face_cx < bottom_right[0]) and (top_left[1] < face_cy < bottom_right[1]):
                        guide_color = (0, 255, 0)
                        alignment_frames += 1
                        instruction = f"Perfect! Auto-focusing... {alignment_frames}/60"
                        
                        if alignment_frames > 60:
                            startX, startY = max(0, int(bbox.origin_x)), max(0, int(bbox.origin_y))
                            endX, endY = min(w, startX + int(bbox.width)), min(h, startY + int(bbox.height))
                            face_crop = frame[startY:endY, startX:endX]
                            
                            user_dir = os.path.join(DB_PATH, safe_name)
                            if not os.path.exists(user_dir):
                                os.makedirs(user_dir)
                                
                            file_path = os.path.join(user_dir, f"{safe_name}_ref.jpg")
                            cv2.imwrite(file_path, face_crop)
                            captured = True
                    else:
                        alignment_frames = 0
                        
                cv2.rectangle(rgb_frame, top_left, bottom_right, guide_color, 3)
                frame_window.image(rgb_frame)
                status_text.markdown(f"**Status:** {instruction}")
                
            cap.release()
            
            if captured:
                frame_window.empty()
                status_text.success(f"✅ Successfully registered {safe_name}! High-res reference saved.")
                try:
                    DeepFace.find(img_path=file_path, db_path=DB_PATH, enforce_detection=False, silent=True)
                except:
                    pass

elif choice == "📊 Log Records":
    st.markdown("""
    <div class="glow-box">
        <h3 style='color:#00FFFF; margin-bottom:0px;'>Daily Attendance Logs</h3>
    </div>
    """, unsafe_allow_html=True)
    
    log_files = [f for f in os.listdir("attendance_logs") if f.endswith(".csv")]
    if len(log_files) == 0:
        st.info("No attendance logs found yet. System is empty.")
    else:
        selected_log = st.selectbox("Select Date to View", log_files)
        df = pd.read_csv(f"attendance_logs/{selected_log}")
        st.dataframe(df, use_container_width=True, height=600)

elif choice == "👥 Registered Users":
    st.markdown("""
    <div class="glow-box">
        <h3 style='color:#00FFFF; margin-bottom:0px;'>Active Database Profiles</h3>
    </div>
    """, unsafe_allow_html=True)
    
    users = [d for d in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, d))]
    
    if not users:
        st.warning("Database is completely empty.")
    else:
        cols = st.columns(4)
        for idx, user in enumerate(users):
            with cols[idx % 4]:
                img_path = os.path.join(DB_PATH, user, f"{user}_ref.jpg")
                img_html = '<h1 style="color:#00FFFF; margin:0;">👤</h1>' 
                
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        b64_str = base64.b64encode(img_file.read()).decode()
                        img_html = f'<div style="width: 90px; height: 90px; margin: 0 auto; border-radius: 50%; overflow: hidden; border: 2px solid #00FFFF; box-shadow: 0 0 10px rgba(0,255,255,0.5);"><img src="data:image/jpeg;base64,{b64_str}" style="width: 100%; height: 100%; object-fit: cover;"></div>'

                st.markdown(f'<div class="glass-card" style="text-align:center; padding: 15px; margin-bottom: 10px;">{img_html}<h5 style="color:#ffffff; margin-top:15px; margin-bottom:0px;">{user.replace("_", " ")}</h5></div>', unsafe_allow_html=True)

elif choice == "🚪 Logout":
    st.session_state["admin_logged_in"] = False
    st.rerun()
