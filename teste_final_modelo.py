import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÃO ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'modelo_pecas_colorido.h5' # muda para o modelo que quer testar sem usar camera
IMAGE_TO_TEST = 'COLOQUE O CAMINHO DA IMAGEM AQUI'

# --- 1. Carregar o Modelo ---
print(f"Carregando modelo: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# --- 2. Carregar e Pré-processar a Imagem ---
img = cv2.imread(IMAGE_TO_TEST)
if img is None:
    print(f"ERRO: Não foi possível carregar a imagem em '{IMAGE_TO_TEST}'")
    exit()


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
img_array = np.expand_dims(img_resized, axis=0)
img_array = img_array / 255.0

# --- 3. Fazer a Previsão ---
prediction = model.predict(img_array)[0][0]

# --- 4. Mostrar Resultado ---
print("\n--- RESULTADO DO TESTE (CORRIGIDO) ---")
print(f"Arquivo: {IMAGE_TO_TEST}")
print(f"Confiança Bruta (0=BOA, 1=DEFEITO): {prediction:.4f}")

if prediction < 0.5:
    print("Previsão: PEÇA BOA")
else:
    print("Previsão: PEÇA COM DEFEITO")