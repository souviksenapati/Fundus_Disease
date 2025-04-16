import os
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from pykalman import KalmanFilter
import tempfile

# ✅ Streamlit UI
st.title("👁️ Glaucoma Detection System (Image & Video)")

# ✅ Auto-select device
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🔍 Using device: **{device.upper()}**")

# ✅ Load model
model_path = "./models/glucoma.pt"
model = YOLO(model_path).to(device)
model.model.float()

# ✅ Kalman Filter
def init_kalman():
    return KalmanFilter(initial_state_mean=0, n_dim_obs=1,
                        transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_covariance=1,
                        observation_covariance=1,
                        transition_covariance=0.01)

kalman_filter = init_kalman()
cdr_estimates = []

# ✅ Frame processing
def process_frame(frame):
    frame = frame.astype(np.float32)
    with torch.no_grad():
        results = model(frame, device=device)

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return None, None, frame.astype(np.uint8)

    cup_box, disc_box = None, None
    cup_conf, disc_conf = 0, 0

    for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        label = "Cup" if cls == 0 else "Disc"
        color = (0, 255, 0) if cls == 0 else (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if cls == 0 and conf > cup_conf:
            cup_box = (x1, y1, x2, y2)
            cup_conf = conf
        elif cls == 1 and conf > disc_conf:
            disc_box = (x1, y1, x2, y2)
            disc_conf = conf

    if cup_conf < 0.3 or disc_conf < 0.3:
        return None, None, frame.astype(np.uint8)

    # ✅ CDR Calculation
    cup_area = (cup_box[2] - cup_box[0]) * (cup_box[3] - cup_box[1])
    disc_area = (disc_box[2] - disc_box[0]) * (disc_box[3] - disc_box[1])
    cup_height = cup_box[3] - cup_box[1]
    disc_height = disc_box[3] - disc_box[1]
    cup_width = cup_box[2] - cup_box[0]
    disc_width = disc_box[2] - disc_box[0]

    cdr = ((cup_area / disc_area) + (cup_height / disc_height) + (cup_width / disc_width)) / 3
    return cdr, (cup_conf + disc_conf) / 2, frame.astype(np.uint8)

# ✅ Full video processing
def calculate_cdr(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_cdr_values = []
    confidence_scores = []

    # Video Writer Setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cdr, confidence, annotated = process_frame(frame)
        out.write(annotated)

        if cdr is not None and confidence is not None:
            frame_cdr_values.append(cdr)
            confidence_scores.append(confidence)

    cap.release()
    out.release()

    if len(frame_cdr_values) > 0:
        cdr_estimates.extend(frame_cdr_values)
        smoothed_cdr_values, _ = kalman_filter.filter(np.array(cdr_estimates))
        smoothed_cdr_values = smoothed_cdr_values.ravel()

        min_len = min(len(smoothed_cdr_values), len(confidence_scores))
        smoothed_cdr_values = smoothed_cdr_values[:min_len]
        confidence_scores = confidence_scores[:min_len]

        final_cdr = np.average(smoothed_cdr_values, weights=confidence_scores)
    else:
        final_cdr = None

    return final_cdr

# ✅ Diagnosis
def get_glaucoma_risk(final_cdr):
    if final_cdr is None:
        return "No valid detections", None

    if final_cdr < 0.4:
        return "No Glaucoma", 0

    elif 0.4 <= final_cdr < 0.5:
        risk_percentage = round(((final_cdr - 0.4) / 0.1) * 100, 1)
        return f"{risk_percentage}% Possibility of Glaucoma", risk_percentage

    else:  # CDR ≥ 0.5
        return "Glaucoma Detected", 100

# ✅ Upload
input_file = st.file_uploader("📂 Upload fundus video or image", type=["mp4", "avi", "jpg", "jpeg", "png"])

if input_file:
    file_type = input_file.type

    if "video" in file_type:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(input_file.read())
            video_path = temp_input.name

        output_path = video_path.replace(".mp4", "_output.mp4")

        with st.spinner("🔍 Analyzing fundus video..."):
            cdr_value = calculate_cdr(video_path, output_path)

        diagnosis, risk = get_glaucoma_risk(cdr_value)

        if cdr_value is not None:
            st.success(f"📊 **Final CDR:** {cdr_value:.3f}")
            st.subheader(f"🩺 **Diagnosis:** {diagnosis}")

            st.video(output_path)
        else:
            st.error("⚠️ No valid CDR detected. Try another input.")

    elif "image" in file_type:
        file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analyzing image..."):
            cdr_value, conf, annotated = process_frame(frame)
            if cdr_value is not None:
                diagnosis, risk = get_glaucoma_risk(cdr_value)
                st.success(f"📊 **Final CDR:** {cdr_value:.3f}")
                st.subheader(f"🩺 **Diagnosis:** {diagnosis}")

                # Convert to RGB for display
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", caption="Annotated Fundus Image")
            else:
                st.error("⚠️ No valid CDR detected. Try another input.")
    else:
        st.error("❌ Unsupported file type.")
