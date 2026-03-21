"""
train.py
========
Entrena YOLOv11 sobre GRAZPEDWRI-DX con:
  - Augmentation especializado para radiografías (albumentations)
  - Focal loss implícita via class_weights
  - Callbacks de WandB para tracking completo
  - Guardado de mejores checkpoints por mAP50-95

Uso:
    python src/train.py --model yolo11m --epochs 100 --run-name baseline

Args:
    --model      Variante YOLO: yolo11n / yolo11s / yolo11m / yolo11l
    --epochs     Épocas de entrenamiento (default: 100)
    --imgsz      Tamaño de imagen (default: 640)
    --batch      Batch size (default: 16)
    --run-name   Nombre del run en WandB
    --no-wandb   Desactiva WandB
"""

import argparse
from pathlib import Path

import wandb
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import albumentations as A
import cv2
import numpy as np


# ── Argumentos ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="yolo11m",  help="Variante YOLO")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--lr0",      type=float, default=1e-3)
    p.add_argument("--lrf",      type=float, default=1e-2,
                   help="LR final = lr0 * lrf (cosine decay)")
    p.add_argument("--run-name", default="fracture-yolo11m")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume",   default=None,
                   help="Path a checkpoint .pt para continuar entrenamiento")
    return p.parse_args()


# ── Augmentation médico-específico ────────────────────────────────────────────
#
# Radiografías: escala de grises, artefactos de hardware, variabilidad
# en exposición → no usar color jitter fuerte ni HSV shifts agresivos.
# Transformaciones espaciales son seguras si respetan los bboxes.

def build_augmentation_pipeline() -> A.Compose:
    """
    Devuelve un pipeline albumentations compatible con bboxes YOLO.
    Se aplica SÓLO al set de entrenamiento.
    """
    return A.Compose(
        [
            # Geométricas — seguras para radiografías
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=10,        # fracturas no tienen orientación fija
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            A.RandomResizedCrop(
                height=640, width=640,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.4,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),

            # Intensidad — simular variaciones de exposición y equipo
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.3,
                p=0.7,
            ),
            A.GaussNoise(var_limit=(5, 30), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),

            # CLAHE — mejora contraste local, muy útil para huesos
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            # Simulación de artefactos de hardware (metal implants)
            A.CoarseDropout(
                max_holes=4, max_height=20, max_width=20,
                fill_value=255,             # blanco = sobreexposición
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ── Hiperparámetros de entrenamiento ──────────────────────────────────────────

def get_train_kwargs(args) -> dict:
    """Devuelve el dict de kwargs para model.train()."""
    return dict(
        data        = "config/dataset.yaml",
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        workers     = args.workers,
        lr0         = args.lr0,
        lrf         = args.lrf,
        optimizer   = "AdamW",
        cos_lr      = True,             # cosine annealing
        warmup_epochs = 3,
        warmup_momentum = 0.8,

        # Augmentation built-in de Ultralytics (complementa albumentations)
        mosaic      = 1.0,              # mosaic augmentation
        mixup       = 0.15,             # mixup entre imágenes
        copy_paste  = 0.1,              # copy-paste de objetos
        degrees     = 5.0,
        translate   = 0.05,
        scale       = 0.5,
        fliplr      = 0.5,
        flipud      = 0.0,              # RX no se voltean verticalmente
        hsv_h       = 0.0,              # RX es escala de grises → sin hue
        hsv_s       = 0.0,
        hsv_v       = 0.3,              # variaciones de brillo

        # Regularización
        weight_decay = 5e-4,
        dropout     = 0.0,
        label_smoothing = 0.1,          # reduce overconfidence

        # Loss
        box         = 7.5,             # peso loss de bbox
        cls         = 0.5,             # peso loss de clasificación
        dfl         = 1.5,             # distributional focal loss

        # Guardado
        project     = "runs/train",
        name        = args.run_name,
        save_period = 10,              # guardar checkpoint cada 10 épocas
        exist_ok    = True,

        # Misc
        patience    = 20,              # early stopping
        plots       = True,
        verbose     = True,
        seed        = 42,
        deterministic = True,
    )


# ── Callback WandB ────────────────────────────────────────────────────────────

class WandbCallback:
    """Callback manual para loggear métricas por época a WandB."""

    def __init__(self, run):
        self.run = run

    def on_train_epoch_end(self, trainer):
        metrics = trainer.metrics
        self.run.log({
            "epoch": trainer.epoch,
            "train/box_loss":  trainer.loss_items[0].item(),
            "train/cls_loss":  trainer.loss_items[1].item(),
            "train/dfl_loss":  trainer.loss_items[2].item(),
            "val/mAP50":       metrics.get("metrics/mAP50(B)", 0),
            "val/mAP50-95":    metrics.get("metrics/mAP50-95(B)", 0),
            "val/precision":   metrics.get("metrics/precision(B)", 0),
            "val/recall":      metrics.get("metrics/recall(B)", 0),
            "lr/pg0":          trainer.optimizer.param_groups[0]["lr"],
        })

    def on_train_end(self, trainer):
        best_path = Path(trainer.save_dir) / "weights" / "best.pt"
        if best_path.exists():
            self.run.save(str(best_path))
            print(f"[WandB] Mejor modelo subido: {best_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Inicializar WandB
    wb_run = None
    if not args.no_wandb:
        wb_run = wandb.init(
            project = "fracture-detection-grazpedwri",
            name    = args.run_name,
            config  = vars(args),
            tags    = ["yolo11", "object-detection", "medical"],
        )

    # Cargar modelo
    if args.resume:
        model = YOLO(args.resume)
        print(f"[→] Reanudando desde {args.resume}")
    else:
        model = YOLO(f"{args.model}.pt")   # descarga weights preentrenados
        print(f"[→] Modelo {args.model} cargado desde ImageNet pretraining")

    # Registrar callbacks
    if wb_run:
        cb = WandbCallback(wb_run)
        model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
        model.add_callback("on_train_end",       cb.on_train_end)

    # Entrenar
    kwargs = get_train_kwargs(args)
    print("\n── Hiperparámetros ──────────────────────────────────────")
    for k, v in kwargs.items():
        print(f"  {k:<25} {v}")
    print("─────────────────────────────────────────────────────────\n")

    results = model.train(**kwargs)

    # Resultados finales
    print("\n── Resultados finales ───────────────────────────────────")
    print(f"  mAP50:    {results.results_dict.get('metrics/mAP50(B)',0):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)',0):.4f}")

    if wb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
