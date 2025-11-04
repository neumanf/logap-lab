import sys
from ultralytics import YOLO


model = YOLO("./resulst/weights/best.pt")

image_path = ""

model(image_path, save=True)