import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import tempfile
import os

# Set up the Streamlit app title
st.title("Drone Tracking and Trailing GUI")

# File uploader for YOLO model
uploaded_model = st.file_uploader("Upload your YOLO model", type=["pt"])

# File uploader for video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi"])

# Check if both files have been uploaded
if uploaded_model and uploaded_video:
    # Save the uploaded model to a temporary file
    model_temp_path = tempfile.NamedTemporaryFile(delete=False)
    model_temp_path.write(uploaded_model.read())
    model_temp_path.close()

    # Load the YOLO model
    model = YOLO(model_temp_path.name)

    # Save the uploaded video to a temporary file
    video_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_temp_path.write(uploaded_video.read())
    video_temp_path.close()

    # Display the video
    st.video(video_temp_path.name)

    # Create a video capture object
    cap = cv2.VideoCapture(video_temp_path.name)
    
    # Store the track history
    track_history = defaultdict(lambda: [])

    # Process the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking on the frame
        results = model.track(source=frame, persist=True)
        if results[0].boxes.id is not None:
            # Get the bounding box coordinates and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:
                    track.pop(0)
                # Draw the tracking lines
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=10)

            # Display the annotated frame in the Streamlit app
            st.image(annotated_frame, channels="BGR")

    cap.release()
    os.remove(model_temp_path.name)
    os.remove(video_temp_path.name)
