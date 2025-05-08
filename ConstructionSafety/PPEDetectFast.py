import cv2
import torch
import time
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# --- 1) Seleção de fonte com tkinter ---
def selecionar_fonte():
    root = tk.Tk()
    root.title("Selecionar Fonte de Vídeo")
    root.geometry("300x120")
    source = {'path': None}

    def usar_webcam():
        source['path'] = 0
        root.destroy()

    def carregar_video():
        filename = filedialog.askopenfilename(
            title="Selecione o arquivo de vídeo",
            filetypes=[("Vídeos MP4/AVI", "*.mp4 *.avi")]
        )
        if filename:
            source['path'] = filename
            root.destroy()

    tk.Button(root, text="Usar Webcam", command=usar_webcam)\
      .pack(fill='x', padx=20, pady=(20,5))
    tk.Button(root, text="Carregar Vídeo", command=carregar_video)\
      .pack(fill='x', padx=20, pady=(0,20))
    root.mainloop()

    if source['path'] is None:
        print("Nenhuma opção selecionada. Encerrando.")
        exit()

    return source['path']

if __name__ == "__main__":
    # 1) Seleciona fonte
    source = selecionar_fonte()

    # 2) Dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    if device == 'cuda':
        print("GPU detectada:", torch.cuda.get_device_name(0))

    # 3) Carrega YOLO e otimiza
    model = YOLO("ppe.pt")
    model.fuse()             
    model.to(device)         
    print("Modelo em:", model.device)

    # 3.1) Classes e PPE correto
    classNames = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
    ]
    ppe_correto = {'Hardhat', 'Mask', 'Safety Vest'}

    # 4) Captura em 256×144
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)

    #  ─── Configura a janela para ser redimensionável ───
    win_name = "PPE Fast Detection"
    # WINDOW_NORMAL permite redimensionar manualmente
    # WINDOW_KEEPRATIO mantém a proporção ao redimensionar
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Desabilita o autosize (para não ajustar à resolução do frame automaticamente)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_AUTOSIZE, 0)
    # Define tamanho inicial
    cv2.resizeWindow(win_name, 320, 180)

    prev_time = 0

    # 5) Loop de inferência
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inferência
        results = model(frame, device=device, half=True, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                cls  = int(box.cls[0])  if hasattr(box.cls,  "__len__") else int(box.cls)
                if conf < 0.5:
                    continue

                label = classNames[cls]
                if label.startswith('NO-'):
                    color = (0, 0, 255)
                elif label in ppe_correto:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        text = f"FPS: {fps:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = frame.shape[1] - w - 10
        y = h + 10
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Só imshow: não mexe mais no tamanho da janela
        cv2.imshow(win_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
