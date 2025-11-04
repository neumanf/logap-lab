from ultralytics import YOLO

# Carrega um modelo pré-treinado de deteção de objetos
model = YOLO("yolo12n.pt")

DATASET_PATH = "./dataset/data.yaml"

results = model.train(
    data=DATASET_PATH,
    epochs=100,
    patience=10,
    batch=2
)
