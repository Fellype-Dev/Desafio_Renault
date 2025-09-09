import cv2
import os
import time
import numpy as np

# --- CONFIGURAÇÃO ---
PASTA_BOAS = "dataset_final/boas"
PASTA_DEFEITO = "dataset_final/com_defeito"
CAMERA_INDEX = 1 # Mude para o índice da sua câmera USB

# Cria as pastas se elas não existirem
os.makedirs(PASTA_BOAS, exist_ok=True)
os.makedirs(PASTA_DEFEITO, exist_ok=True)
# --------------------

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a câmera no índice {CAMERA_INDEX}.")
    exit()

print("\n--- Coleta de Dataset Avançada (Salva apenas a ROI) ---")
print("Pressione 'b' para salvar a tampa detectada como 'BOA'.")
print("Pressione 'd' para salvar a tampa detectada como 'COM DEFEITO'.")
print("Pressione 'q' para sair.")
print("----------------------------------------------------------\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_feedback = frame.copy()
    
    # deteccao de circulo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=20, maxRadius=100)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Só processa se um círculo for encontrado
    if circles is not None:
        circles = np.uint16(np.around(circles))
        c = circles[0, 0]
        center = (c[0], c[1])
        radius = c[2]
        
        # Desenha o círculo na imagem de feedback para sabermos que ele foi detectado
        cv2.circle(frame_feedback, center, radius, (0, 255, 0), 3)

        # Se uma tecla de captura for pressionada
        if key == ord('b') or key == ord('d'):
            # Recorta a região de interesse (ROI)
            x, y, r = c
            start_x = max(x - r, 0)
            end_x = min(x + r, frame.shape[1])
            start_y = max(y - r, 0)
            end_y = min(y + r, frame.shape[0])
            
            crop_roi = frame[start_y:end_y, start_x:end_x]
            
            if crop_roi.size > 0:
                timestamp = int(time.time() * 1000)
                if key == ord('b'):
                    nome_arquivo = f"boa_{timestamp}.jpg"
                    caminho_salvar = os.path.join(PASTA_BOAS, nome_arquivo)
                    cv2.imwrite(caminho_salvar, crop_roi)
                    print(f"Imagem 'BOA' (recortada) salva: {caminho_salvar}")
                
                elif key == ord('d'):
                    nome_arquivo = f"defeito_{timestamp}.jpg"
                    caminho_salvar = os.path.join(PASTA_DEFEITO, nome_arquivo)
                    cv2.imwrite(caminho_salvar, crop_roi)
                    print(f"Imagem 'COM DEFEITO' (recortada) salva: {caminho_salvar}")
                
                time.sleep(0.2) # Pequena pausa para não salvar várias imagens de uma vez

    cv2.imshow("Coleta Avancada - Aperte 'b' ou 'd' quando o circulo estiver verde", frame_feedback)

cap.release()
cv2.destroyAllWindows()