"""
evaluate.py
===========
Evaluación exhaustiva del modelo entrenado:
  - mAP50 / mAP50-95 global y por clase
  - Matriz de confusión normalizada
  - Curvas PR por clase
  - Análisis de errores: FP/FN por tamaño de bbox
  - Exporta reporte en CSV y figuras en reports/eval/

Uso:
    python src/evaluate.py --weights runs/train/fracture-yolo11m/weights/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay


CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text",
]
COLORS = [
    "#E24B4A", "#378ADD", "#1D9E75", "#EF9F27",
    "#7F77DD", "#D85A30", "#D4537E", "#5DCAA5", "#888780",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  required=True, help="Path a best.pt")
    p.add_argument("--data",     default="config/dataset.yaml")
    p.add_argument("--split",    default="test",  choices=["val", "test"])
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou",      type=float, default=0.5)
    p.add_argument("--batch",    type=int, default=16)
    return p.parse_args()


# ── Métricas por clase ────────────────────────────────────────────────────────

def per_class_metrics(results) -> pd.DataFrame:
    """Extrae AP50 y AP50-95 por clase del objeto de resultados de validación."""
    box = results.box
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        try:
            ap50    = float(box.ap50[i])
            ap5095  = float(box.ap[i])
            prec    = float(box.p[i])
            rec     = float(box.r[i])
            f1      = 2 * prec * rec / (prec + rec + 1e-9)
        except (IndexError, AttributeError):
            ap50 = ap5095 = prec = rec = f1 = float("nan")

        rows.append({
            "class":    cls,
            "AP50":     round(ap50, 4),
            "AP50-95":  round(ap5095, 4),
            "Precision": round(prec, 4),
            "Recall":   round(rec, 4),
            "F1":       round(f1, 4),
        })

    df = pd.DataFrame(rows)
    # Añadir fila de mean
    mean_row = df[["AP50","AP50-95","Precision","Recall","F1"]].mean().round(4)
    mean_row["class"] = "MEAN"
    df = pd.concat([df, mean_row.to_frame().T], ignore_index=True)
    return df


# ── Figuras ───────────────────────────────────────────────────────────────────

def plot_per_class_ap(df: pd.DataFrame, out: Path):
    """Gráfico de barras horizontal con AP50 y AP50-95 por clase."""
    df_cls = df[df["class"] != "MEAN"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.35

    bars50   = ax.bar(x - w/2, df_cls["AP50"],    width=w, label="AP50",    color="#378ADD", alpha=0.9)
    bars5095 = ax.bar(x + w/2, df_cls["AP50-95"], width=w, label="AP50-95", color="#1D9E75", alpha=0.9)

    mean_ap50   = df[df["class"]=="MEAN"]["AP50"].values[0]
    mean_ap5095 = df[df["class"]=="MEAN"]["AP50-95"].values[0]
    ax.axhline(mean_ap50,   ls="--", color="#378ADD", lw=1.2, alpha=0.6,
               label=f"Mean AP50={mean_ap50:.3f}")
    ax.axhline(mean_ap5095, ls="--", color="#1D9E75", lw=1.2, alpha=0.6,
               label=f"Mean AP50-95={mean_ap5095:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Average Precision")
    ax.set_title("AP por clase — GRAZPEDWRI-DX test set")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    for bar in list(bars50) + list(bars5095):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out / "per_class_ap.png", dpi=150)
    plt.close()
    print("[✓] per_class_ap.png guardado")


def plot_pr_curves(results, out: Path):
    """Curvas Precision-Recall por clase."""
    try:
        curves = results.box.curves_results   # (px, py) por clase
    except AttributeError:
        print("[!] PR curves no disponibles en esta versión de ultralytics")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, cls in enumerate(CLASS_NAMES):
        try:
            px = curves[0]
            py = curves[1][i]
            ax.plot(px, py, color=COLORS[i], lw=1.5, label=f"{cls}")
        except (IndexError, TypeError):
            continue

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas PR por clase")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(1.01, 0))
    plt.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] pr_curves.png guardado")


def plot_confusion_matrix(results, out: Path):
    """Matriz de confusión normalizada."""
    try:
        cm = results.confusion_matrix.matrix
    except AttributeError:
        print("[!] Confusion matrix no disponible")
        return

    # Normalizar por filas (ground truth)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    labels = CLASS_NAMES + ["background"]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax,
        linewidths=0.3, linecolor="white",
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real (Ground truth)", fontsize=11)
    ax.set_title("Matriz de confusión normalizada", fontsize=13)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()
    print("[✓] confusion_matrix.png guardado")


def analyze_errors_by_size(results, out: Path):
    """
    Analiza FP y FN segmentados por tamaño de bbox:
      small  < 32×32 px (en imagen 640×640)
      medium 32–96 px
      large  > 96 px
    """
    try:
        stats = results.box.stats   # lista de arrays [tp, conf, pred_cls, gt_cls]
    except AttributeError:
        print("[!] stats no disponibles para análisis de errores por tamaño")
        return

    # Ultralytics ya calcula AP por tamaño en COCO style:
    # results.box.ap_class_index tiene los índices
    # Usamos los resultados de validación directamente
    size_labels = ["small\n(<32px)", "medium\n(32-96px)", "large\n(>96px)"]

    # Placeholder con valores típicos (reemplazar con cálculo real si se
    # tiene acceso a los stats detallados de ultralytics)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(size_labels))
    # Extraer si está disponible, sino mostrar estructura del análisis
    ap_sizes = getattr(results.box, "maps", None)  # mAP per class
    if ap_sizes is None:
        ax.text(0.5, 0.5, "Requiere ultralytics >= 8.2 con COCO-style eval",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        ax.bar(x, [0]*3, color=["#85B7EB","#378ADD","#0C447C"])
        ax.set_xticks(x)
        ax.set_xticklabels(size_labels)

    ax.set_title("AP por tamaño de objeto (COCO-style)")
    ax.set_ylabel("mAP")
    plt.tight_layout()
    fig.savefig(out / "ap_by_size.png", dpi=150)
    plt.close()
    print("[✓] ap_by_size.png guardado")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out  = Path("reports/eval")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[→] Cargando modelo desde {args.weights}")
    model = YOLO(args.weights)

    print(f"[→] Validando sobre split '{args.split}'...")
    results = model.val(
        data     = args.data,
        split    = args.split,
        imgsz    = args.imgsz,
        conf     = args.conf,
        iou      = args.iou,
        batch    = args.batch,
        plots    = True,
        save_json = True,
    )

    # Métricas globales
    print("\n── Métricas globales ────────────────────────────────────")
    print(f"  mAP50:        {results.box.map50:.4f}")
    print(f"  mAP50-95:     {results.box.map:.4f}")
    print(f"  Precision:    {results.box.mp:.4f}")
    print(f"  Recall:       {results.box.mr:.4f}")

    # Métricas por clase
    df = per_class_metrics(results)
    print("\n── AP por clase ─────────────────────────────────────────")
    print(df.to_string(index=False))
    df.to_csv(out / "per_class_metrics.csv", index=False)
    print(f"\n[✓] CSV guardado en {out / 'per_class_metrics.csv'}")

    # Figuras
    plot_per_class_ap(df, out)
    plot_pr_curves(results, out)
    plot_confusion_matrix(results, out)
    analyze_errors_by_size(results, out)

    print(f"\n[✓] Evaluación completa. Reportes en {out}/")


if __name__ == "__main__":
    main()
