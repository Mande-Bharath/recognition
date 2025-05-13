import cv2
import torch
import pyttsx3
import time
import numpy as np
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Smooth and natural speech
engine.setProperty('volume', 1.0)  # Max volume

# Select the best available device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model (Extra-large version for best accuracy)
model = YOLO("yolov8x.pt").to(device)

# Open webcam with optimized settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Track last spoken commands
last_speech_time_dict = {}

# Camera Calibration for Distance Estimation
KNOWN_WIDTHS = {
    "person": 0.45,  # Avg human shoulder width in meters
    "car": 1.8,  # Avg car width in meters
    "bottle": 0.07,  # Avg bottle width in meters
    "chair": 0.5,  # Avg chair width
}
FOCAL_LENGTH = 800  # Estimated focal length (can be calibrated)

def estimate_distance(label, pixel_width):
    """Estimate real-world distance of an object based on bounding box width."""
    if label in KNOWN_WIDTHS and pixel_width > 0:
        return (KNOWN_WIDTHS[label] * FOCAL_LENGTH) / pixel_width
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform object detection with optimized settings
    results = model(frame, imgsz=1280, conf=0.5, iou=0.4)  

    frame_width = frame.shape[1]
    object_positions = []

    # Extract detected objects, positions, and sizes
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())  
            label = model.names[cls_id]  
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)  
            pixel_width = int(box.xyxy[0][2] - box.xyxy[0][0])  # Width of bounding box

            distance = estimate_distance(label, pixel_width)  # Calculate estimated distance

            if distance is not None:
                distance_text = f"{distance:.2f} meters"
                object_positions.append((label, x_center, distance_text, distance))

    # Process multiple objects and determine safest path
    current_time = time.time()
    command = ""
    
    if object_positions:
        # Sort objects by closest distance
        object_positions.sort(key=lambda x: x[3])

        for obj_label, obj_x, obj_distance_text, obj_distance in object_positions:
            # Generate movement command based on object position
            if obj_x < frame_width // 3:
                command = f"{obj_label} on the left, {obj_distance_text}, move right"
            elif obj_x > 2 * frame_width // 3:
                command = f"{obj_label} on the right, {obj_distance_text}, move left"
            else:
                command = f"{obj_label} ahead, {obj_distance_text}, stop"

            # Speak command only if 4 seconds have passed for this object
            if obj_label not in last_speech_time_dict or (current_time - last_speech_time_dict[obj_label]) > 4:
                print(command)
                engine.say(command)
                engine.runAndWait()
                last_speech_time_dict[obj_label] = current_time  # Update timestamp

            # Speak only the **most critical** object to avoid repetition
            break  

    # Annotate frame with bounding boxes and distances
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Navigation for Blind - Ultra Accurate", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()






