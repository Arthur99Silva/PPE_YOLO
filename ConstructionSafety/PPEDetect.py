# PPEDetect_gui.py
from ultralytics import YOLO
import cv2
import cvzone
import math
import PySimpleGUI as sg

# --- 1) Janela de seleção de fonte ---
sg.theme('DarkBlue3')
layout = [
    [sg.Text('Escolha a fonte de vídeo para detecção de EPI')],
    [sg.Button('Usar Webcam'), sg.Button('Carregar Vídeo')],
    [sg.Button('Cancelar')]
]
window = sg.Window('PPE Detector', layout)

event, values = window.read()
window.close()

if event == 'Usar Webcam':
    source = 0
elif event == 'Carregar Vídeo':
    # abre um diálogo de seleção de arquivo
    source = sg.popup_get_file(
        'Selecione o arquivo de vídeo',
        file_types=(("Vídeos MP4/AVI", "*.mp4;*.avi"),),
        no_window=True
    )
    if not source:
        sg.popup_error('Nenhum vídeo selecionado. Encerrando.')
        exit()
else:
    exit()

# --- 2) Inicializa captura ---
cap = cv2.VideoCapture(source)
cap.set(3, 1280)  # opcional: largura
cap.set(4, 720)   # opcional: altura

# --- 3) Carrega o modelo e classes ---
model = YOLO("ppe.pt")
classNames = [
    'Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest',
    'Person','Safety Cone','Safety Vest','machinery','vehicle'
]

# --- 4) Loop de processamento ---
while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            label = classNames[cls]

            # cor conforme presença/ausência de EPI
            if conf > 0.5:
                if label.startswith('NO-'):
                    color = (0, 0, 255)
                elif label in ['Hardhat','Safety Vest','Mask']:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cvzone.putTextRect(
                    img, f'{label} {conf}',
                    (max(0, x1), max(35, y1)),
                    scale=1, thickness=1,
                    colorB=color, colorT=(255,255,255),
                    colorR=color, offset=5
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("PPE Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
