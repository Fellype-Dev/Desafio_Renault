import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Configuração ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'modelo_pecas_teste.h5' # modelo que vai querer usar no teste (caso treine um modelo novo e queria testar, é só mudar aqui)
CAMERA_INDEX = 1 # indice da camera usb (depende de quantas cameras tiver no pc)

print(f"Carregando modelo: {MODEL_PATH}...")
try:
    classification_model = load_model(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Inicializa a câmera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a câmera no índice {CAMERA_INDEX}.")
    exit()

print("\n--- Inspeção Avançada (CORRIGIDA) Iniciada ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resultado = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        c = circles[0, 0]
        center = (c[0], c[1])
        radius = c[2]

        
        x, y, r = c
        start_x = max(x - r, 0)
        end_x = min(x + r, frame.shape[1])
        start_y = max(y - r, 0)
        end_y = min(y + r, frame.shape[0])
        

        crop_roi = frame[start_y:end_y, start_x:end_x]
        
        if crop_roi.size > 0:
            roi_rgb = cv2.cvtColor(crop_roi, cv2.COLOR_BGR2RGB)

            # Pré-processa a ROI para o modelo
            img_resized = cv2.resize(roi_rgb, (IMG_WIDTH, IMG_HEIGHT))
            img_array = np.expand_dims(img_resized, axis=0)
            img_array = img_array / 255.0

            # Faz a previsão
            prediction = classification_model.predict(img_array, verbose=0)[0][0]

            if prediction < 0.5:
                texto = "Status: BOA"
                cor = (0, 255, 0) # Verde
            else:
                texto = "Status: COM DEFEITO"
                cor = (0, 0, 255) # Vermelho
            
            # Desenha os resultados na imagem
            cv2.circle(frame_resultado, center, radius, cor, 3)
            cv2.putText(frame_resultado, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
            cv2.putText(frame_resultado, f"Confianca: {prediction:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

    else:
        cv2.putText(frame_resultado, "Tampa nao encontrada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
    cv2.imshow("Inspecao Avancada e Corrigida", frame_resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()