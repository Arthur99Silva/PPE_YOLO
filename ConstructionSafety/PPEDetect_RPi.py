import argparse
import math
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta EPIs (Raspberry Pi / CPU-only)"
    )
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="0 para webcam ou caminho para vídeo (mp4/avi)"
    )
    parser.add_argument(
        "--weights", "-w",
        default="ppe.pt",
        help="Caminho para o modelo YOLO (.pt)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Threshold de confiança (0–1)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        help="Device: 'cpu' (no Pi) ou '0','1'… para GPU"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1) Fonte de vídeo ---
    try:
        src = int(args.source)
    except ValueError:
        src = args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- 2) Carrega modelo no CPU ---
    model = YOLO(args.weights)
    # força CPU-only:
    model.to(args.device)

    # --- 3) Classes e cores ---
    classNames = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
    ]
    # cores BGR: verde=OK, vermelho=ausente, azul=outros
    colors = {
        **{c: (0,255,0) for c in ['Hardhat','Mask','Safety Vest']},
        **{f"NO-{c}": (0,0,255) for c in ['Hardhat','Mask','Safety Vest']},
    }

    # --- 4) Loop de detecção ---
    while True:
        ret, img = cap.read()
        if not ret: break

        # retorna um Results; pegamos o primeiro
        results = model(img, device=args.device)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < args.conf: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = classNames[cls]

            color = colors.get(label, (255,0,0))
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        cv2.imshow("PPE Detection", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
