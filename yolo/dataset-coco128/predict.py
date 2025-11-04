from ultralytics import YOLO

# Caminho do modelo treinado
MODEL_PATH = "./runs/detect/train/weights/best.pt"

# Caminho da imagem. Opções disponíveis: https://docs.ultralytics.com/modes/predict/#inference-sources 
IMAGE_PATH = "https://www.vmcdn.ca/f/files/victoriatimescolonist/json/2022/03/web1_vka-viewstreet-13264.jpg"

model = YOLO(MODEL_PATH)

results = model(IMAGE_PATH) # ou [IMAGEM_1, IMAGEM_2]

for result in results:
    result.show()
    result.save(filename="resultado.jpg")