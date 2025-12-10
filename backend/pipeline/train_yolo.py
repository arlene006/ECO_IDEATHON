from ultralytics import YOLO

# Load YOLOv8 model (choose nano, small, medium as needed)
model = YOLO("yolov8s.pt")   # good balance between speed & accuracy

# Train
model.train(
    data="datasets_solar/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",    # Use GPU if available, else CPU
    name="solar_detector"
)
