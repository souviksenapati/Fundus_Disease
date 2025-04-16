import os
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from pykalman import KalmanFilter

# Create output directory
output_dir = "D:/Projects/Fundus_Disease"
os.makedirs(output_dir, exist_ok=True)

# Streamlit App Title
st.title("Glaucoma Detection with YOLOv8")

# Video Input
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file is not None:
    video_path = os.path.join(output_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{input_filename}_output.avi")

    # Load YOLOv8 model in float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("D:/Projects/Fundus_Disease/glucoma.pt").to(device)
    model.model.float()

    # Initialize Kalman Filter
    def init_kalman():
        return KalmanFilter(initial_state_mean=0, n_dim_obs=1, 
                            transition_matrices=[1], 
                            observation_matrices=[1], 
                            initial_state_covariance=1, 
                            observation_covariance=1, 
                            transition_covariance=0.01)

    kalman_filter = init_kalman()
    cdr_estimates = []

    # Process a single frame
    def process_frame(frame):
        frame = frame.astype(np.float32)
        with torch.no_grad():
            results = model(frame)

        cup_box, disc_box = None, None
        cup_conf, disc_conf = 0, 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box

                if cls == 0 and conf > cup_conf:
                    cup_box = (x1, y1, x2, y2)
                    cup_conf = conf
                elif cls == 1 and conf > disc_conf:
                    disc_box = (x1, y1, x2, y2)
                    disc_conf = conf

        if cup_conf < 0.5 or disc_conf < 0.5:
            return None, None

        cup_area = (cup_box[2] - cup_box[0]) * (cup_box[3] - cup_box[1])
        disc_area = (disc_box[2] - disc_box[0]) * (disc_box[3] - disc_box[1])
        cup_height = cup_box[3] - cup_box[1]
        disc_height = disc_box[3] - disc_box[1]
        cup_width = cup_box[2] - cup_box[0]
        disc_width = disc_box[2] - disc_box[0]

        cdr = ((cup_area / disc_area) + (cup_height / disc_height) + (cup_width / disc_width)) / 3
        return cdr, (cup_conf + disc_conf) / 2

    # Process entire video sequentially
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

        if len(frame_cdr_values) > 0:
            cdr_estimates.extend(frame_cdr_values)
            smoothed_cdr_values, _ = kalman_filter.filter(np.array(cdr_estimates))
            smoothed_cdr_values = smoothed_cdr_values.ravel()
            min_length = min(len(smoothed_cdr_values), len(confidence_scores))
            smoothed_cdr_values = smoothed_cdr_values[:min_length]
            confidence_scores = confidence_scores[:min_length]
            final_cdr = np.average(smoothed_cdr_values, weights=confidence_scores)
        else:
            final_cdr = None

        if final_cdr is None:
            diagnosis = "No valid detections"
        elif final_cdr < 0.4:
            diagnosis = "No Glaucoma"
        elif 0.4 <= final_cdr <= 0.5:
            diagnosis = "Possible Glaucoma (Needs Further Tests)"
        else:
            diagnosis = "High Risk of Glaucoma"

        return final_cdr, diagnosis

    # Run YOLOv8-Based CDR Calculation
    cdr_value, result = calculate_cdr(video_path)

    # Print Final Results
    st.write(f"Final CDR: {cdr_value:.3f}" if cdr_value else "No valid CDR detected")
    st.write(f"Diagnosis: {result}")
