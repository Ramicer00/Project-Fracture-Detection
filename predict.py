"""
predict.py
==========
Inferencia sobre imágenes nuevas con:
  - Predicción con umbral configurable
  - Grad-CAM sobre el backbone (EigenCAM)
  - Visualización de boxes + heatmap superpuesto
  - Exportación del modelo a ONNX para deploy

Uso:
    # Predicción sobre una imagen
    python src/predict.py --weights best.pt --source radiografia.jpg

    # Sobre un directorio
    python src/predict.py --weights best.pt --source data/test/images/

    # Exportar a ONNX
    python src/predict.py --weights best.pt --export-onnx
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch

from ultralytics import YOLO


CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text",
]
COLORS_BGR = [
    (66,  75,  226),   # boneanomaly  → rojo
    (221, 138,  55),   # bonelesion   → azul
    (117, 158,  29),   # foreignbody  → verde
    (39,  239, 167),   # fracture     → cian
    (221, 119, 127),   # metal        → lavanda
    (48,  90,  216),   # periosteal   → naranja
    (126,  83, 212),   # pronatorsign → rosa
    (165, 202,  93),   # softtissue   → teal
    (128, 135, 136),   # text         → gris
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",      default="runs/train/fracture-yolo11m/weights/best.pt")
    p.add_argument("--source",       default=None,
                   help="Imagen, directorio o 0 para webcam")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.5)
    p.add_argument("--imgsz",        type=int,   default=640)
    p.add_argument("--save-dir",     default="runs/predict")
    p.add_argument("--show-gradcam", action="store_true",
                   help="Genera mapa Grad-CAM (requiere pytorch-grad-cam)")
    p.add_argument("--export-onnx",  action="store_true",
                   help="Exporta el modelo a ONNX y sale")
    return p.parse_args()


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(model: YOLO, img_path: str) -> np.ndarray:
    """
    Aplica EigenCAM sobre la última capa convolucional del backbone.
    Devuelve un heatmap uint8 del mismo tamaño que la imagen.

    EigenCAM no requiere gradientes reales → es más rápido y estable
    que Grad-CAM++ para modelos de detección.
    """
    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[!] Instalá pytorch-grad-cam: pip install grad-cam")
        return None

    # Extraer el backbone de PyTorch (ultralytics expone model.model)
    torch_model = model.model
    torch_model.eval()

    # Target layer: última capa C2f del backbone (índice -3 típicamente)
    # Ajustar según la variante de YOLO
    target_layer = None
    for name, module in torch_model.model.named_modules():
        if "C2f" in type(module).__name__:
            target_layer = module
    if target_layer is None:
        print("[!] No se encontró la capa target para Grad-CAM")
        return None

    cam = EigenCAM(model=torch_model, target_layers=[target_layer])

    # Preprocesar imagen
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Redimensionar y normalizar para el modelo
    img_resized = cv2.resize(img_rgb, (640, 640))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)

    # Calcular CAM
    grayscale_cam = cam(input_tensor=tensor)
    grayscale_cam = grayscale_cam[0]

    # Superponer en imagen original
    img_float = img_resized.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    visualization = cv2.resize(visualization, (w, h))

    return visualization


# ── Visualización de predicciones ─────────────────────────────────────────────

def draw_predictions(img: np.ndarray, boxes, confs, classes) -> np.ndarray:
    """Dibuja bounding boxes con etiquetas sobre la imagen."""
    out = img.copy()
    for box, conf, cls_id in zip(boxes, confs, classes):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def visualize_and_save(
    img_path: str,
    result,
    gradcam: np.ndarray | None,
    save_dir: Path,
):
    """Guarda la imagen con boxes y (opcionalmente) Grad-CAM side-by-side."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] No se pudo leer {img_path}")
        return

    boxes   = result.boxes.xyxy.cpu().numpy()   if result.boxes else np.empty((0,4))
    confs   = result.boxes.conf.cpu().numpy()    if result.boxes else np.empty(0)
    classes = result.boxes.cls.cpu().numpy()     if result.boxes else np.empty(0)

    pred_img = draw_predictions(img, boxes, confs, classes)
    stem = Path(img_path).stem

    if gradcam is not None:
        # Lado a lado: predicción | Grad-CAM
        pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.imshow(pred_rgb)
        ax1.set_title("Predicciones del modelo", fontsize=13)
        ax1.axis("off")
        ax2.imshow(gradcam)
        ax2.set_title("EigenCAM — zonas de atención del modelo", fontsize=13)
        ax2.axis("off")

        # Leyenda de clases detectadas
        detected = list(set(int(c) for c in classes))
        patches = [
            mpatches.Patch(
                color=tuple(v/255 for v in COLORS_BGR[c][::-1]),
                label=CLASS_NAMES[c],
            )
            for c in detected
        ]
        if patches:
            ax1.legend(handles=patches, loc="lower left", fontsize=9,
                       framealpha=0.8)

        plt.tight_layout()
        out_path = save_dir / f"{stem}_gradcam.png"
        fig.savefig(out_path, dpi=150)
        plt.close()
    else:
        out_path = save_dir / f"{stem}_pred.jpg"
        cv2.imwrite(str(out_path), pred_img)

    print(f"[✓] Guardado: {out_path} ({len(boxes)} detecciones)")


# ── Export ONNX ───────────────────────────────────────────────────────────────

def export_to_onnx(model: YOLO, imgsz: int):
    """Exporta el modelo a ONNX con dynamic batch support."""
    print("[→] Exportando a ONNX...")
    path = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        dynamic  = True,      # batch dinámico para deployment
        simplify = True,      # onnx-simplifier
        opset    = 17,
    )
    print(f"[✓] ONNX exportado en: {path}")
    print("    Para inferencia rápida en CPU:")
    print("    import onnxruntime as ort")
    print(f"    sess = ort.InferenceSession('{path}')")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    print(f"[✓] Modelo cargado desde {args.weights}")

    if args.export_onnx:
        export_to_onnx(model, args.imgsz)
        return

    if args.source is None:
        print("[!] Especificá --source <imagen/directorio>")
        return

    # Recolectar imágenes
    source = Path(args.source)
    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
    else:
        print(f"[!] Fuente no encontrada: {source}")
        return

    print(f"[→] Procesando {len(images)} imagen(es)...")

    for img_path in images:
        # Predicción
        results = model.predict(
            source   = str(img_path),
            conf     = args.conf,
            iou      = args.iou,
            imgsz    = args.imgsz,
            verbose  = False,
        )
        result = results[0]

        # Grad-CAM (opcional)
        gradcam = None
        if args.show_gradcam:
            gradcam = compute_gradcam(model, str(img_path))

        visualize_and_save(str(img_path), result, gradcam, save_dir)

    print(f"\n[✓] Resultados guardados en {save_dir}/")


if __name__ == "__main__":
    main()
