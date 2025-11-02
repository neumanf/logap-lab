from ultralytics import YOLO


model = YOLO("./runs/detect/train/weights/best.pt")

image_path = "path/to/bus.jpg"

model(image_path, save=True)