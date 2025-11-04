import os
from ultralytics import YOLO

model = YOLO('yolo12n.pt')

data_dir = os.path.join("dataset", "data.yaml")

train_result = model.train(
    data=data_dir, 
    epochs=50, 
    batch=10
)
