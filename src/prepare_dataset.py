"""
prepare_dataset.py
==================
Descarga GRAZPEDWRI-DX desde Kaggle, verifica integridad,
genera EDA básico y muestra estadísticas de anotaciones.

Uso:
    python src/prepare_dataset.py

Requisito: ~/.kaggle/kaggle.json con tus credenciales.
"""

import os
import json
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm


# ── Configuración ────────────────────────────────────────────────────────────

ROOT        = Path("data/GRAZPEDWRI-DX")
IMAGES_DIR  = ROOT / "images"
LABELS_DIR  = ROOT / "labels"

CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text",
]
COLORS = [
    "#E24B4A", "#378ADD", "#1D9E75", "#EF9F27",
    "#7F77DD", "#D85A30", "#D4537E", "#5DCAA5", "#888780",
]


# ── Descarga ─────────────────────────────────────────────────────────────────

def download_dataset():
    """Descarga el dataset desde Kaggle si no existe."""
    if ROOT.exists() and any(ROOT.iterdir()):
        print(f"[✓] Dataset ya existe en {ROOT}")
        return

    print("[↓] Descargando GRAZPEDWRI-DX desde Kaggle...")
    ROOT.mkdir(parents=True, exist_ok=True)
    os.system(
        "kaggle datasets download -d cokane53/grazpedwri-dx "
        f"--unzip -p {ROOT}"
    )
    print("[✓] Descarga completada")


# ── EDA ──────────────────────────────────────────────────────────────────────

def parse_labels(split: str) -> pd.DataFrame:
    """Lee todas las anotaciones YOLO de un split y devuelve DataFrame."""
    label_path = LABELS_DIR / split
    rows = []
    for lf in label_path.glob("*.txt"):
        img_path = IMAGES_DIR / split / lf.with_suffix(".jpg").name
        try:
            w, h = Image.open(img_path).size
        except Exception:
            w, h = 1, 1

        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                rows.append({
                    "split": split,
                    "file": lf.stem,
                    "class_id": int(cls),
                    "class_name": CLASS_NAMES[int(cls)],
                    "cx": cx, "cy": cy,
                    "bw": bw, "bh": bh,
                    "area": bw * bh,
                    "img_w": w, "img_h": h,
                    "abs_area": bw * bh * w * h,
                })
    return pd.DataFrame(rows)


def run_eda():
    """Genera figuras de EDA y las guarda en reports/."""
    out = Path("reports/eda")
    out.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    dfs = {s: parse_labels(s) for s in splits if (LABELS_DIR / s).exists()}
    df  = pd.concat(dfs.values(), ignore_index=True)

    # 1. Distribución de clases ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    bars = axes[0].bar(CLASS_NAMES, counts.values,
                       color=[COLORS[i] for i in range(len(CLASS_NAMES))],
                       edgecolor="none", width=0.7)
    axes[0].set_title("Distribución de clases (todas las anotaciones)", fontsize=13)
    axes[0].set_ylabel("Cantidad de bounding boxes")
    axes[0].tick_params(axis="x", rotation=40)
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{v:,}", ha="center", va="bottom", fontsize=9)

    # Proporción por split
    split_counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    split_counts.T.plot(kind="bar", ax=axes[1], colormap="Set2",
                        edgecolor="none", width=0.75)
    axes[1].set_title("Anotaciones por clase y split", fontsize=13)
    axes[1].set_ylabel("Cantidad")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].legend(title="Split")

    plt.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close()
    print("[✓] class_distribution.png guardado")

    # 2. Tamaño de bounding boxes (pequeños vs grandes) ───────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(CLASS_NAMES):
        subset = df[df["class_name"] == cls]["area"]
        if len(subset) == 0:
            continue
        ax.scatter(
            np.random.normal(i, 0.15, len(subset)),
            subset * 100,   # en % del área de imagen
            alpha=0.3, s=8, color=COLORS[i], label=cls,
        )
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
    ax.set_ylabel("Área del bbox (% del área de imagen)")
    ax.set_title("Distribución de tamaño de bounding boxes por clase")
    ax.axhline(1.0, ls="--", color="gray", lw=0.8, label="1% área")
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=8, markerscale=2)
    plt.tight_layout()
    fig.savefig(out / "bbox_sizes.png", dpi=150)
    plt.close()
    print("[✓] bbox_sizes.png guardado")

    # 3. Heatmap de centros de bbox ────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, cls, color in zip(axes.flat, CLASS_NAMES, COLORS):
        subset = df[df["class_name"] == cls]
        if len(subset) == 0:
            ax.set_title(cls); ax.axis("off"); continue
        heatmap, _, _ = np.histogram2d(
            subset["cx"], subset["cy"], bins=40, range=[[0,1],[0,1]]
        )
        ax.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto",
                  extent=[0,1,0,1])
        ax.set_title(f"{cls} (n={len(subset):,})", fontsize=9)
        ax.set_xlabel("cx"); ax.set_ylabel("cy")

    plt.suptitle("Distribución espacial de centros de bbox", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out / "bbox_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] bbox_heatmaps.png guardado")

    # 4. Resumen estadístico ───────────────────────────────────────────────
    summary = df.groupby("class_name").agg(
        n_boxes=("class_id", "count"),
        mean_area_pct=("area", lambda x: f"{x.mean()*100:.2f}%"),
        median_area_pct=("area", lambda x: f"{x.median()*100:.2f}%"),
    ).reindex(CLASS_NAMES)

    print("\n── Resumen de anotaciones ──────────────────────────────")
    print(summary.to_string())
    summary.to_csv(out / "annotation_summary.csv")

    # 5. Imbalance ratio ───────────────────────────────────────────────────
    max_cls = counts.max()
    print("\n── Imbalance ratio (vs clase mayoritaria) ──────────────")
    for cls in CLASS_NAMES:
        ratio = max_cls / max(counts[cls], 1)
        print(f"  {cls:<20} {ratio:>6.1f}×")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_dataset()
    run_eda()
    print("\n[✓] EDA completado. Revisá reports/eda/")
