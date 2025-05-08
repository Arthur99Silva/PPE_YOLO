# PPEDetect_sem_gui.py
from ultralytics import YOLO
import cv2
import cvzone
import math
import tkinter as tk
from tkinter import filedialog

# --- 1) Interface de seleção de fonte (via tkinter) ---
def escolher_fonte():
    print("Selecione a fonte:")
    print("1 - Usar Webcam")
    print("2 - Carregar Vídeo")
    escolha = input("Digite 1 ou 2: ")

    if escolha == '1':
        return 0  # Webcam
    elif escolha == '2':
        tk.Tk().withdraw()  # Oculta a janela principal do Tkinter
        filename = filedialog.askopenfilename(
            title='Selecione o vídeo',
            filetypes=[("Vídeos MP4/AVI", "*.mp4 *.avi")]
        )
        if not filename:
            print("Nenhum vídeo selecionado. Encerrando.")
            exit()
        return filename
    else:
        print("Opção inválida. Encerrando.")
        exit()

source = escolher_fonte()

# --- 2) Inicializa captura ---
cap = cv2.VideoCapture(source)
cap.set(3, 1280)
cap.set(4, 720)

# --- 3) Carrega modelo e classes ---
model = YOLO("ppe.pt")
classNames = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

# --- 4) Loop de detecção ---
while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]

            if conf > 0.5:
                if label.startswith('NO-'):
                    color = (0, 0, 255)
                elif label in ['Hardhat', 'Safety Vest', 'Mask']:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cvzone.putTextRect(
                    img, f'{label} {conf}',
                    (max(0, x1), max(35, y1)),
                    scale=1, thickness=1,
                    colorB=color, colorT=(255, 255, 255),
                    colorR=color, offset=5
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("PPE Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
