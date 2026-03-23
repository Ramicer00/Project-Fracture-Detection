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


---

## Estructura del proyecto

```
fracture-detection/
├── app.py                   # Demo
├── requirements.txt
├── config/
│   └── dataset.yaml         # Config YOLO con paths y class weights
├── src/
│   ├── prepare_dataset.py   # Descarga, EDA, estadísticas de anotaciones
│   ├── train.py             # Entrenamiento YOLOv11 con WandB
│   ├── evaluate.py          # Evaluación exhaustiva + figuras
│   ├── predict.py           # Inferencia + Grad-CAM + export ONNX
│   └── benchmark.py         # Benchmark YOLOv11 vs RT-DETR
└── data/                    # Dataset

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

```

---

### Entrenamiento

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

---

### Evaluación

```bash
python src/evaluate.py \
    --weights runs/train/baseline-yolo11m/weights/best.pt \
    --split test
```


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

### Inferencia y Grad-CAM

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

### Benchmark YOLOv11 vs RT-DETR

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

### Demo interactiva

```bash
python app.py
# → Abre en http://localhost:7860
```

---

## Próximos pasos

- [ ] **TTA (Test-Time Augmentation):** promediar predicciones sobre versiones
      aumentadas de cada imagen → +1-2 pp de mAP sin reentrenar
- [ ] **Ensemble:** combinar YOLOv11m + YOLOv11l con Weighted Boxes Fusion (WBF)
- [ ] **Semi-supervised:** usar imágenes sin etiquetar del dataset extendido
      con pseudo-labels generados por el modelo
- [ ] **Análisis clínico:** comparar errores del modelo con tasa de error
      humana reportada en el paper original del dataset

---

## Referencias

- Dataset: Nagy et al., "GRAZPEDWRI-DX — Pediatric Wrist Fracture X-ray Dataset", 2022
- YOLOv11: Ultralytics YOLO11 Architecture
- RT-DETR: "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024
- EigenCAM: "Eigen-CAM: Class Activation Map using Principal Components", 2020
