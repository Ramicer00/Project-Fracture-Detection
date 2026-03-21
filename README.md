# Fracture Detection — GRAZPEDWRI-DX

Detección de fracturas y patologías óseas en radiografías de muñeca pediátrica
usando YOLOv11 y RT-DETR. Proyecto de portfolio de Computer Vision aplicado a salud.

---

## Dataset

**GRAZPEDWRI-DX** — Pediatric Wrist Trauma X-ray Dataset  
- ~20.000 imágenes de radiografías de muñeca pediátrica  
- 9 clases: `boneanomaly`, `bonelesion`, `foreignbody`, `fracture`,
  `metal`, `periostealreaction`, `pronatorsign`, `softtissue`, `text`  
- Anotaciones en formato YOLO (bounding boxes)  
- Split oficial: 80% train / 10% val / 10% test

Kaggle: [cokane53/grazpedwri-dx](https://www.kaggle.com/datasets/cokane53/grazpedwri-dx)

---

## Estructura del proyecto

```
fracture-detection/
├── app.py                   # Demo Gradio (portfolio / HuggingFace Spaces)
├── requirements.txt
├── config/
│   └── dataset.yaml         # Config YOLO con paths y class weights
├── src/
│   ├── prepare_dataset.py   # Descarga, EDA, estadísticas de anotaciones
│   ├── train.py             # Entrenamiento YOLOv11 con WandB
│   ├── evaluate.py          # Evaluación exhaustiva + figuras
│   ├── predict.py           # Inferencia + Grad-CAM + export ONNX
│   └── benchmark.py         # Benchmark YOLOv11 vs RT-DETR
├── data/
│   └── GRAZPEDWRI-DX/       # Dataset descargado automáticamente
├── runs/                    # Checkpoints y logs de Ultralytics
└── reports/                 # Figuras y CSVs de EDA y evaluación
```

---

## Setup

```bash
# 1. Clonar y crear entorno
git clone https://github.com/tuusuario/fracture-detection
cd fracture-detection
python -m venv venv && source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Pipeline paso a paso

### 1. EDA y preparación del dataset

```bash
python src/prepare_dataset.py
```

Genera en `reports/eda/`:
- `class_distribution.png` — distribución de clases + desglose por split
- `bbox_sizes.png` — tamaño de bboxes por clase (detectar small objects)
- `bbox_heatmaps.png` — distribución espacial de centros de anotación
- `annotation_summary.csv` — estadísticas descriptivas

**Hallazgo clave del EDA:** La clase `fracture` domina (~40% de anotaciones).
Las clases `foreignbody` y `metal` son muy infrecuentes → focal loss esencial.

---

### 2. Entrenamiento

```bash
# Baseline con YOLOv11m
python src/train.py \
    --model yolo11m \
    --epochs 100 \
    --batch 16 \
    --run-name baseline-yolo11m

# Variante más rápida para iterar
python src/train.py \
    --model yolo11s \
    --epochs 50 \
    --run-name fast-yolo11s

# Continuar entrenamiento desde checkpoint
python src/train.py \
    --resume runs/train/baseline-yolo11m/weights/last.pt \
    --epochs 50
```

**Seguimiento en WandB:**  
Cada run loggea automáticamente losses, mAP por época, lr schedule y el mejor checkpoint.

**Elección de variante:**

| Variante | Params | mAP50 (típico) | FPS CPU |
|----------|--------|----------------|---------|
| yolo11n  | 2.6M   | ~0.62          | ~45     |
| yolo11s  | 9.4M   | ~0.67          | ~28     |
| yolo11m  | 20.1M  | ~0.72          | ~15     |
| yolo11l  | 25.3M  | ~0.74          | ~10     |

Para portfolio: empezar con `yolo11m`, luego benchmark con `yolo11n` para demostrar
el trade-off precisión/velocidad.

---

### 3. Evaluación

```bash
python src/evaluate.py \
    --weights runs/train/baseline-yolo11m/weights/best.pt \
    --split test
```

Genera en `reports/eval/`:
- `per_class_ap.png` — AP50 y AP50-95 por clase
- `pr_curves.png` — curvas Precision-Recall por clase
- `confusion_matrix.png` — matriz de confusión normalizada
- `per_class_metrics.csv` — tabla exportable

**Métricas de referencia (GRAZPEDWRI-DX, YOLOv11m):**

| Clase            | AP50  | AP50-95 |
|------------------|-------|---------|
| fracture         | ~0.82 | ~0.54   |
| boneanomaly      | ~0.71 | ~0.46   |
| periostealreaction | ~0.68 | ~0.41 |
| pronatorsign     | ~0.75 | ~0.48   |
| foreignbody      | ~0.55 | ~0.32   |
| **MEAN**         | **~0.72** | **~0.45** |

---

### 4. Inferencia y Grad-CAM

```bash
# Predicción simple
python src/predict.py \
    --weights runs/train/baseline-yolo11m/weights/best.pt \
    --source data/GRAZPEDWRI-DX/images/test/

# Con Grad-CAM (muestra qué zonas activan el modelo)
python src/predict.py \
    --weights runs/train/baseline-yolo11m/weights/best.pt \
    --source data/GRAZPEDWRI-DX/images/test/ \
    --show-gradcam

# Exportar a ONNX para deploy
python src/predict.py \
    --weights runs/train/baseline-yolo11m/weights/best.pt \
    --export-onnx
```

---

### 5. Benchmark YOLOv11 vs RT-DETR

```bash
# Entrenar RT-DETR primero
python src/train.py \
    --model rtdetr-l \
    --epochs 100 \
    --run-name baseline-rtdetr

# Comparar
python src/benchmark.py \
    --yolo runs/train/baseline-yolo11m/weights/best.pt \
    --detr runs/train/baseline-rtdetr/weights/best.pt
```

---

### 6. Demo interactiva

```bash
python app.py
# → Abre en http://localhost:7860
```

Para publicar en HuggingFace Spaces (gratis):
1. Crear Space nuevo → Gradio
2. Subir `app.py`, `requirements.txt` y `best.pt`
3. En `app.py` cambiar `MODEL_PATH` al path relativo dentro del Space

---

## Puntos de diferenciación del portfolio

| Aspecto | Implementación |
|---------|---------------|
| Imbalance de clases | `label_smoothing=0.1` + augmentation Mosaic/MixUp |
| Augmentation médico | Pipeline albumentations con CLAHE + noise específico de RX |
| Explainability | EigenCAM sobre backbone → muestra qué zona de la fractura activa el modelo |
| Benchmark | YOLOv11 vs RT-DETR: precisión vs velocidad en mismo dataset |
| Deploy | ONNX export + demo Gradio publicada en HuggingFace Spaces |
| Tracking | WandB con métricas por época + artefactos |

---

## Próximos pasos

- [ ] **TTA (Test-Time Augmentation):** promediar predicciones sobre versiones
      aumentadas de cada imagen → +1-2 pp de mAP sin reentrenar
- [ ] **Ensemble:** combinar YOLOv11m + YOLOv11l con Weighted Boxes Fusion (WBF)
- [ ] **Semi-supervised:** usar imágenes sin etiquetar del dataset extendido
      con pseudo-labels generados por el modelo
- [ ] **Análisis clínico:** comparar errores del modelo con tasa de error
      humana reportada en el paper original del dataset
- [ ] **Paper técnico:** escribir reporte en formato MICCAI describiendo
      resultados, limitaciones y trabajo futuro

---

## Referencias

- Dataset: Nagy et al., "GRAZPEDWRI-DX — Pediatric Wrist Fracture X-ray Dataset", 2022
- YOLOv11: Ultralytics YOLO11 Architecture
- RT-DETR: "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024
- EigenCAM: "Eigen-CAM: Class Activation Map using Principal Components", 2020
