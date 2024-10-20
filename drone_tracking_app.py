import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Set up the Streamlit app title
st.title("Drone Tracking and Trailing GUI")

# Load the YOLO model
model = YOLO("C:\\Users\\Asus\\Downloads\\best (1).pt")

# Display the video
video_path = "C:\\Users\\Asus\\Downloads\\videoplayback (1).mp4"

if video_path:
    # Create a video capture object
    cap = cv2.VideoCapture(video_path)
    
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
