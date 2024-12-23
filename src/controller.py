import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

# Load a model
model = YOLO("../models/best.pt")  # load an official model

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

move_locker = "dummy"

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x , y, c = frame.shape

    results = model.track(frame)  # predict on an image

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

    annotated_frame = results[0].plot()

    cv2.imshow('YOLO V8 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break


# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

