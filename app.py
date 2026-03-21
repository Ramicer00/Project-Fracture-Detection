"""
app.py
======
Demo interactiva con Gradio para el portfolio.
Subís una radiografía de muñeca y el modelo detecta
las patologías con boxes, confianza y (opcional) Grad-CAM.

Uso local:
    python app.py

Deploy en HuggingFace Spaces:
    Subir este archivo como app.py con requirements.txt

Dependencia adicional:
    pip install gradio>=4.0
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from ultralytics import YOLO


# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_PATH  = "runs/train/fracture-yolo11m/weights/best.pt"
CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text",
]
COLORS = [
    (226, 75,  66),   # boneanomaly
    (55,  138, 221),  # bonelesion
    (29,  158, 117),  # foreignbody
    (167, 239,  39),  # fracture
    (127, 119, 221),  # metal
    (216,  90,  48),  # periostealreaction
    (212,  83, 126),  # pronatorsign
    (93,  202, 165),  # softtissue
    (136, 135, 128),  # text
]
CLINICAL_INFO = {
    "boneanomaly":      "Anomalía ósea estructural.",
    "bonelesion":       "Lesión en el tejido óseo. Requiere evaluación adicional.",
    "foreignbody":      "Objeto extraño detectado.",
    "fracture":         "Fractura ósea detectada. Evaluar con especialista.",
    "metal":            "Implante o material metálico.",
    "periostealreaction":"Reacción perióstica. Posible proceso inflamatorio o neoplásico.",
    "pronatorsign":     "Signo pronador. Indicador de fractura no desplazada del radio.",
    "softtissue":       "Anomalía en tejidos blandos.",
    "text":             "Anotación de texto en la imagen.",
}

model = None   # lazy load


def load_model():
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}. "
                "Ejecutá primero: python src/train.py"
            )
        model = YOLO(MODEL_PATH)
    return model


# ── Inferencia ────────────────────────────────────────────────────────────────

def predict(image: np.ndarray, conf_thresh: float, iou_thresh: float) -> tuple:
    """
    Corre el modelo sobre la imagen y devuelve:
      - imagen con boxes dibujados (RGB np.ndarray)
      - texto con resumen de hallazgos
    """
    m = load_model()

    # Guardar temporalmente (Ultralytics acepta np.array directamente)
    results = m.predict(
        source  = image,
        conf    = conf_thresh,
        iou     = iou_thresh,
        imgsz   = 640,
        verbose = False,
    )[0]

    # Dibujar predicciones
    out_img = image.copy()
    findings = {}

    if results.boxes and len(results.boxes):
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            color = COLORS[cls_id]
            cls_name = CLASS_NAMES[cls_id]

            # Dibujar box
            cv2.rectangle(out_img, (x1, y1), (x2, y2),
                          color[::-1], 2)   # PIL es RGB, cv2 es BGR

            # Label
            label = f"{cls_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(out_img,
                          (x1, y1 - th - 8), (x1 + tw + 4, y1),
                          color[::-1], -1)
            cv2.putText(out_img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            if cls_name not in findings or findings[cls_name] < conf:
                findings[cls_name] = conf

    # Resumen textual de hallazgos
    if findings:
        lines = ["## Hallazgos detectados\n"]
        for cls_name, conf in sorted(findings.items(),
                                     key=lambda x: -x[1]):
            info = CLINICAL_INFO.get(cls_name, "")
            lines.append(
                f"**{cls_name}** — confianza: {conf:.0%}  \n{info}\n"
            )
        lines.append(
            "\n> ⚠️ Este modelo es una herramienta de investigación y no "
            "reemplaza el diagnóstico médico profesional."
        )
        report = "\n".join(lines)
    else:
        report = "No se detectaron hallazgos con el umbral actual."

    return out_img, report


# ── Interfaz Gradio ───────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Fracture Detection — GRAZPEDWRI-DX") as demo:

        gr.Markdown(
            "# Detección de fracturas en radiografías pediátricas\n"
            "**Modelo:** YOLOv11m entrenado en GRAZPEDWRI-DX · 9 clases · mAP50 ≈ 0.72\n\n"
            "Subí una radiografía de muñeca pediátrica y el modelo detectará "
            "fracturas y otras patologías."
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(
                    type="numpy",
                    label="Radiografía de entrada",
                    height=400,
                )
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Umbral de confianza",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Umbral IoU (NMS)",
                    )
                btn = gr.Button("Detectar patologías", variant="primary")

            with gr.Column(scale=1):
                out_img = gr.Image(
                    type="numpy",
                    label="Detecciones",
                    height=400,
                )
                out_report = gr.Markdown(label="Hallazgos")

        btn.click(
            fn      = predict,
            inputs  = [inp_img, conf_slider, iou_slider],
            outputs = [out_img, out_report],
        )

        # Auto-submit al cambiar sliders
        conf_slider.change(predict, [inp_img, conf_slider, iou_slider],
                           [out_img, out_report])
        iou_slider.change( predict, [inp_img, conf_slider, iou_slider],
                           [out_img, out_report])

        gr.Markdown(
            "### Clases detectadas\n"
            "| Clase | Descripción |\n|---|---|\n" +
            "\n".join(f"| `{k}` | {v} |" for k, v in CLINICAL_INFO.items())
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,    # True para link público temporal
    )
