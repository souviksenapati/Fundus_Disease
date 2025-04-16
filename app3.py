import os
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from pykalman import KalmanFilter

# ✅ Streamlit UI
st.title("👁️ Glaucoma Detection with YOLOv8")

# ✅ Automatically select device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🔍 Using device: **{device.upper()}**")

# ✅ Load YOLOv8 model
model_path = "D:/Projects/Fundus_Disease/glucoma.pt"  # Update with your model path
model = YOLO(model_path).to(device)
model.model.float()  # Ensure float32 precision

# ✅ Create output directory
output_dir = "D:/Projects/Fundus_Disease"
os.makedirs(output_dir, exist_ok=True)

# ✅ Upload Video File
video_file = st.file_uploader("📂 Upload a fundus video", type=["mp4", "avi"])

# ✅ Initialize Kalman Filter
def init_kalman():
    return KalmanFilter(initial_state_mean=0, n_dim_obs=1, 
                        transition_matrices=[1], 
                        observation_matrices=[1], 
                        initial_state_covariance=1, 
                        observation_covariance=1, 
                        transition_covariance=0.01)

kalman_filter = init_kalman()
cdr_estimates = []  # Stores smoothed CDR values

# ✅ Process a single frame
def process_frame(frame):
    frame = frame.astype(np.float32)  # Ensure input is float32
    
    with torch.no_grad():  # Disable gradient calculation for faster inference
        results = model(frame, device=device)  # ✅ Use detected device (GPU or CPU)

    cup_box, disc_box = None, None
    cup_conf, disc_conf = 0, 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box

            if cls == 0 and conf > cup_conf:  # Optic Cup
                cup_box = (x1, y1, x2, y2)
                cup_conf = conf
            elif cls == 1 and conf > disc_conf:  # Optic Disc
                disc_box = (x1, y1, x2, y2)
                disc_conf = conf

    if cup_conf < 0.5 or disc_conf < 0.5:
        return None, None

    # ✅ Compute CDR using area, height, and width ratios
    cup_area = (cup_box[2] - cup_box[0]) * (cup_box[3] - cup_box[1])
    disc_area = (disc_box[2] - disc_box[0]) * (disc_box[3] - disc_box[1])
    cup_height = cup_box[3] - cup_box[1]
    disc_height = disc_box[3] - disc_box[1]
    cup_width = cup_box[2] - cup_box[0]
    disc_width = disc_box[2] - disc_box[0]

    cdr = ((cup_area / disc_area) + (cup_height / disc_height) + (cup_width / disc_width)) / 3
    return cdr, (cup_conf + disc_conf) / 2  # Return CDR and confidence score

# ✅ Process entire video
def calculate_cdr(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_cdr_values = []
    confidence_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cdr, confidence = process_frame(frame)
        if cdr is not None and confidence is not None:
            frame_cdr_values.append(cdr)
            confidence_scores.append(confidence)

    cap.release()

    # ✅ Apply Kalman Filter Smoothing
    if len(frame_cdr_values) > 0:
        cdr_estimates.extend(frame_cdr_values)
        smoothed_cdr_values, _ = kalman_filter.filter(np.array(cdr_estimates))

        # ✅ Flatten smoothed values to 1D
        smoothed_cdr_values = smoothed_cdr_values.ravel()  

        # ✅ Ensure same length before computing weighted average
        min_length = min(len(smoothed_cdr_values), len(confidence_scores))
        smoothed_cdr_values = smoothed_cdr_values[:min_length]
        confidence_scores = confidence_scores[:min_length]

        final_cdr = np.average(smoothed_cdr_values, weights=confidence_scores)
    else:
        final_cdr = None

    return final_cdr

# ✅ Glaucoma Detection Logic with fine-grained risk percentage
def get_glaucoma_risk(final_cdr):
    if final_cdr is None:
        return "No valid detections", None
    
    if final_cdr < 0.4:
        return "No Glaucoma", 0
    
    elif 0.4 <= final_cdr < 0.5:
        # Gradually increase risk from 10% to 100%
        risk_percentage = round(((final_cdr - 0.4) / 0.1) * 100, 1)
        return f"{risk_percentage}% Possibility of Glaucoma", risk_percentage
    
    else:  # CDR ≥ 0.5
        risk_percentage = round(min(100 + (final_cdr - 0.5) * 100, 200), 1)
        return f"{risk_percentage}% Possibility of High-Risk Glaucoma", risk_percentage

# ✅ If video uploaded, process it
if video_file:
    video_path = os.path.join(output_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # ✅ Run YOLOv8-Based CDR Calculation
    with st.spinner("🔍 Analyzing Fundus Video..."):
        cdr_value = calculate_cdr(video_path)

    # ✅ Get Final Diagnosis
    diagnosis, risk = get_glaucoma_risk(cdr_value)

    # ✅ Display Results
    if cdr_value is not None:
        st.success(f"📊 **Final CDR:** {cdr_value:.3f}")
        st.subheader(f"🩺 **Diagnosis:** {diagnosis}")
    else:
        st.error("⚠️ No valid CDR detected. Try another video.")
