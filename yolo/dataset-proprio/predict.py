from ultralytics import YOLO

# Caminho do modelo treinado
MODEL_PATH = "./runs/detect/train/weights/best.pt"

# Caminho da imagem
IMAGE_PATH = "./imagem.jpg"

model = YOLO(MODEL_PATH)

results = model(IMAGE_PATH)

for result in results:
    result.show()
    result.save(filename="resultado.jpg")