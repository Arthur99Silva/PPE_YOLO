# Passo 1 - Importação das bibliotecas
from ultralytics import YOLO
import cv2
import cvzone
import math
 
# Passo 2 - Carregando o vídeo
#cap = cv2.VideoCapture(0)  # Webcam
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture("C:/Users/Arthur/Documents/YOLO_Learning/PPE_YOLO/PPE_YOLO/ConstructionSafety/Videos/ppe-3.mp4")  # Video

# Passo 3 - Carregando o modelo
model = YOLO("ppe.pt")
 
# Passo 4 - Definição das classes de detecção
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

# Passo 5 - Loop principal de preocessamento
while True:
    success, img = cap.read()
    results = model(img, stream=True)

# Passo 6 - Processamento das detecções
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
# Passo 7 - Confiança da detecção
            conf = math.ceil((box.conf[0] * 100)) / 100

# Passo 8 - Identificando a classe detectada
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

# Passo 9 - Colorindo as detecções
            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)

# Passo 10 - Colorindo caixas e textos
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

# Passo 11 - Exibição do resultado
    cv2.imshow("Image", img)
    cv2.waitKey(1)