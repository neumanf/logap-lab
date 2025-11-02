from ultralytics import YOLO

model = YOLO("yolo12n.pt")

results = model.train(
    data="coco128.yaml", # Dataset contendo as primeiras 128 imagens do dataset COCO 2017
    epochs=100, # Cada época representa uma passagem completa por todo o conjunto de dados
    patience=10, # Número de épocas sem melhoria nas métricas de validação antes de interromper antecipadamente o treinamento
)
