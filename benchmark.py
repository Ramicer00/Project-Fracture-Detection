"""
benchmark.py
============
Compara YOLOv11 vs RT-DETR en el mismo test set:
  - mAP50 / mAP50-95
  - Latencia en CPU y GPU (ms/imagen)
  - Throughput (FPS)
  - Parámetros y tamaño del modelo

Uso:
    python src/benchmark.py \
        --yolo  runs/train/fracture-yolo11m/weights/best.pt \
        --detr  runs/train/fracture-rtdetr/weights/best.pt

Si solo tenés uno de los dos modelos, el script igualmente corre
y genera el reporte parcial.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO, RTDETR


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo",    default=None, help="Path a best.pt de YOLOv11")
    p.add_argument("--detr",    default=None, help="Path a best.pt de RT-DETR")
    p.add_argument("--data",    default="config/dataset.yaml")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--n-runs",  type=int, default=100,
                   help="Iteraciones para medir latencia")
    p.add_argument("--batch",   type=int, default=1,
                   help="Batch size para benchmark de latencia")
    return p.parse_args()


# ── Medición de latencia ──────────────────────────────────────────────────────

def measure_latency(model, imgsz: int, n_runs: int, device: str) -> dict:
    """
    Mide latencia promedio (warm-up de 10 runs, luego n_runs).
    Devuelve dict con latencia media, std y FPS.
    """
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model.model(dummy)

    # Medición
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return {
        "mean_ms":  round(lat.mean(), 2),
        "std_ms":   round(lat.std(),  2),
        "p95_ms":   round(np.percentile(lat, 95), 2),
        "fps":      round(1000 / lat.mean(), 1),
    }


def count_params(model) -> dict:
    """Cuenta parámetros y tamaño aproximado del modelo."""
    total = sum(p.numel() for p in model.model.parameters())
    train = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)   # float32
    return {
        "total_params":  f"{total/1e6:.1f}M",
        "trainable":     f"{train/1e6:.1f}M",
        "size_mb":       round(size_mb, 1),
    }


# ── Evaluación de mAP ─────────────────────────────────────────────────────────

def evaluate_map(model, data: str, imgsz: int) -> dict:
    results = model.val(data=data, imgsz=imgsz, verbose=False)
    return {
        "mAP50":    round(results.box.map50, 4),
        "mAP50-95": round(results.box.map,   4),
        "precision": round(results.box.mp,   4),
        "recall":    round(results.box.mr,   4),
    }


# ── Comparativa visual ────────────────────────────────────────────────────────

def plot_benchmark(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    models = df["model"].tolist()
    colors = ["#378ADD", "#1D9E75"][:len(models)]

    # mAP
    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    for i, (_, row) in enumerate(df.iterrows()):
        ax.bar(x - w/2 + i*w,
               [row["mAP50"], row["mAP50-95"]],
               w, label=row["model"], color=colors[i], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(["mAP50", "mAP50-95"])
    ax.set_ylim(0, 1); ax.set_title("Precisión")
    ax.legend(); ax.set_ylabel("AP")

    # Latencia
    ax = axes[1]
    lats = df["mean_ms"].tolist()
    stds = df["std_ms"].tolist()
    bars = ax.bar(models, lats, color=colors, alpha=0.85, width=0.5,
                  yerr=stds, capsize=6)
    ax.set_title("Latencia (CPU, ms/imagen)")
    ax.set_ylabel("ms")
    for bar, v in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v}ms", ha="center", fontsize=10, fontweight="bold")

    # FPS
    ax = axes[2]
    bars = ax.bar(models, df["fps"].tolist(), color=colors, alpha=0.85, width=0.5)
    ax.set_title("Throughput (FPS en CPU)")
    ax.set_ylabel("FPS")
    for bar, v in zip(bars, df["fps"].tolist()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("YOLOv11 vs RT-DETR — GRAZPEDWRI-DX", fontsize=14, y=1.02)
    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figura guardada en {out / 'benchmark_comparison.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out    = Path("reports/benchmark")

    rows = []
    model_configs = []
    if args.yolo:
        model_configs.append(("YOLOv11m", YOLO, args.yolo))
    if args.detr:
        model_configs.append(("RT-DETR-l", RTDETR, args.detr))

    if not model_configs:
        print("[!] Especificá al menos --yolo o --detr")
        return

    for name, ModelClass, weights in model_configs:
        print(f"\n── Evaluando {name} ─────────────────────────────────")
        model = ModelClass(weights)
        model.model.to(device)
        model.model.eval()

        params  = count_params(model)
        latency = measure_latency(model, args.imgsz, args.n_runs, device)
        accuracy = evaluate_map(model, args.data, args.imgsz)

        row = {"model": name, **accuracy, **latency, **params}
        rows.append(row)

        print(f"  mAP50:     {accuracy['mAP50']}")
        print(f"  mAP50-95:  {accuracy['mAP50-95']}")
        print(f"  Latencia:  {latency['mean_ms']} ± {latency['std_ms']} ms")
        print(f"  FPS:       {latency['fps']}")
        print(f"  Parámetros:{params['total_params']}  ({params['size_mb']} MB)")

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark_results.csv", index=False)
    print(f"\n[✓] CSV guardado en {out / 'benchmark_results.csv'}")
    print("\n── Resumen ──────────────────────────────────────────────")
    print(df[["model","mAP50","mAP50-95","mean_ms","fps","total_params"]].to_string(index=False))

    if len(rows) >= 2:
        plot_benchmark(df, out)


if __name__ == "__main__":
    main()
