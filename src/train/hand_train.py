from ultralytics import YOLO
# Load a model
model = YOLO("../../models/yolo11n-pose.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Train the model
    results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)