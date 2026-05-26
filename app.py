import os

# =========================================================
# FIX FOR KERAS / TENSORFLOW COMPATIBILITY
# =========================================================
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
import zipfile

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Attendance System",
    layout="wide"
)

# =========================================================
# DATABASE FOLDERS
# =========================================================
DB_PATH = "registered_users"

for folder in [DB_PATH, "attendance_logs"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# =========================================================
# DOWNLOAD MEDIAPIPE MODEL
# =========================================================
model_path = "blaze_face_short_range.tflite"

if not os.path.exists(model_path):

    st.info("Downloading MediaPipe model...")

    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        model_path
    )

# =========================================================
# LOAD AI MODELS
# =========================================================
@st.cache_resource
def load_ai_models():

    # =========================================
    # EXTRACT ZIP MODEL
    # =========================================
    if not os.path.exists("robust_mask_detector.h5"):

        if os.path.exists("robust_mask_detector.zip"):

            with zipfile.ZipFile(
                "robust_mask_detector.zip",
                "r"
            ) as zip_ref:

                zip_ref.extractall(".")

        else:
            st.error("robust_mask_detector.zip not found!")
            return None, None

    # =========================================
    # LOAD MASK MODEL
    # =========================================
    try:

        mask_model = load_model(
            "robust_mask_detector.h5",
            compile=False
        )

        st.success("✅ Mask model loaded successfully!")

    except Exception as e:

        st.error(f"Model loading error: {e}")

        mask_model = None

    # =========================================
    # LOAD FACE DETECTOR
    # =========================================
    try:

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.7
        )

        face_detector = vision.FaceDetector.create_from_options(
            options
        )

    except Exception as e:

        st.error(f"Face detector loading error: {e}")

        face_detector = None

    # =========================================
    # PRELOAD DEEPFACE
    # =========================================
    try:
        DeepFace.build_model("Facenet")
    except:
        pass

    return mask_model, face_detector

# =========================================================
# LOAD MODELS
# =========================================================
mask_model, face_detector = load_ai_models()

# =========================================================
# UI
# =========================================================
st.title("SMART ATTENDANCE SYSTEM")

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Live Attendance",
        "Register User",
        "Attendance Logs",
        "Registered Users"
    ]
)

# =========================================================
# HOME
# =========================================================
if menu == "Home":

    st.subheader("AI Powered Attendance System")

    st.write("""
    Features:
    - Face Recognition
    - Mask Detection
    - Attendance Logging
    - DeepFace Verification
    - MediaPipe Face Detection
    """)

# =========================================================
# LIVE ATTENDANCE
# =========================================================
elif menu == "Live Attendance":

    registered = [
        f for f in os.listdir(DB_PATH)
        if os.path.isdir(os.path.join(DB_PATH, f))
    ]

    if len(registered) == 0:

        st.error("No registered users found!")

    else:

        st.subheader("Live Attendance Scanner")

        run = st.button("Start Scanner")

        FRAME_WINDOW = st.image([])

        if run:

            cap = cv2.VideoCapture(0)

            while True:

                ret, frame = cap.read()

                if not ret:
                    st.error("Camera not working")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                detection_result = face_detector.detect(
                    mp_image
                )

                mask_status = "Unknown"
                identity_name = "Unknown"

                if detection_result.detections:

                    detection = detection_result.detections[0]

                    bbox = detection.bounding_box

                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    w = int(bbox.width)
                    h = int(bbox.height)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    face_crop = frame[y:y+h, x:x+w]

                    if face_crop.size != 0:

                        # =====================================
                        # MASK PREDICTION
                        # =====================================
                        if mask_model is not None:

                            try:

                                rgb_face = cv2.cvtColor(
                                    face_crop,
                                    cv2.COLOR_BGR2RGB
                                )

                                resized_face = cv2.resize(
                                    rgb_face,
                                    (224, 224)
                                )

                                img_array = np.expand_dims(
                                    resized_face,
                                    axis=0
                                )

                                preprocessed_img = preprocess_input(
                                    img_array
                                )

                                prediction = mask_model.predict(
                                    preprocessed_img
                                )[0][0]

                                mask_status = (
                                    "Mask ON"
                                    if prediction < 0.5
                                    else "No Mask"
                                )

                            except Exception as e:
                                st.error(e)

                        # =====================================
                        # FACE RECOGNITION
                        # =====================================
                        try:

                            cv2.imwrite(
                                "temp_scan.jpg",
                                face_crop
                            )

                            dfs = DeepFace.find(
                                img_path="temp_scan.jpg",
                                db_path=DB_PATH,
                                model_name="Facenet",
                                enforce_detection=False,
                                silent=True
                            )

                            if (
                                len(dfs) > 0
                                and not dfs[0].empty
                            ):

                                match_path = dfs[0].iloc[0]["identity"]

                                identity_name = os.path.basename(
                                    os.path.dirname(match_path)
                                ).replace("_", " ")

                        except:
                            pass

                        # =====================================
                        # SAVE ATTENDANCE
                        # =====================================
                        date_string = datetime.now().strftime(
                            "%Y-%m-%d"
                        )

                        time_string = datetime.now().strftime(
                            "%H:%M:%S"
                        )

                        log_file = (
                            f"attendance_logs/log_{date_string}.csv"
                        )

                        log_entry = pd.DataFrame({
                            "Name": [identity_name],
                            "Time": [time_string],
                            "Mask Status": [mask_status]
                        })

                        if not os.path.isfile(log_file):

                            log_entry.to_csv(
                                log_file,
                                index=False
                            )

                        else:

                            log_entry.to_csv(
                                log_file,
                                mode="a",
                                header=False,
                                index=False
                            )

                        cv2.putText(
                            frame,
                            identity_name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            mask_status,
                            (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0),
                            2
                        )

                FRAME_WINDOW.image(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                )

# =========================================================
# REGISTER USER
# =========================================================
elif menu == "Register User":

    st.subheader("Register New User")

    new_name = st.text_input("Enter Name")

    if st.button("Capture Face"):

        if new_name == "":

            st.warning("Enter a name first")

        else:

            cap = cv2.VideoCapture(0)

            ret, frame = cap.read()

            if ret:

                user_dir = os.path.join(
                    DB_PATH,
                    new_name.replace(" ", "_")
                )

                if not os.path.exists(user_dir):
                    os.makedirs(user_dir)

                file_path = os.path.join(
                    user_dir,
                    f"{new_name}_ref.jpg"
                )

                cv2.imwrite(file_path, frame)

                st.success("User registered successfully!")

                st.image(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                )

            cap.release()

# =========================================================
# ATTENDANCE LOGS
# =========================================================
elif menu == "Attendance Logs":

    st.subheader("Attendance Logs")

    log_files = [
        f for f in os.listdir("attendance_logs")
        if f.endswith(".csv")
    ]

    if len(log_files) == 0:

        st.warning("No logs found")

    else:

        selected_log = st.selectbox(
            "Select Log",
            log_files
        )

        df = pd.read_csv(
            f"attendance_logs/{selected_log}"
        )

        st.dataframe(df)

# =========================================================
# REGISTERED USERS
# =========================================================
elif menu == "Registered Users":

    st.subheader("Registered Users")

    users = [
        d for d in os.listdir(DB_PATH)
        if os.path.isdir(os.path.join(DB_PATH, d))
    ]

    if not users:

        st.warning("No users registered")

    else:

        cols = st.columns(4)

        for idx, user in enumerate(users):

            with cols[idx % 4]:

                img_path = os.path.join(
                    DB_PATH,
                    user,
                    f"{user}_ref.jpg"
                )

                if os.path.exists(img_path):

                    st.image(
                        img_path,
                        width=150
                    )

                st.write(user.replace("_", " "))
```
