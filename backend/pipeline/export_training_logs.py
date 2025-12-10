import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = r"runs/detect/solar_detector"
CSV_PATH = os.path.join(LOG_DIR, "results.csv")
OUT_DIR = r"training_logs"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# -------------------------------------------------------
# 1. Save a clean training metrics CSV
# -------------------------------------------------------
df.to_csv(os.path.join(OUT_DIR, "training_metrics.csv"), index=False)
print("✔ Saved: training_metrics.csv")


# -------------------------------------------------------
# 2. LOSS CURVE (train + val)
# -------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["train/box_loss"], label="Train Box Loss")
plt.plot(df["train/cls_loss"], label="Train Class Loss")
plt.plot(df["train/dfl_loss"], label="Train DFL Loss")

plt.plot(df["val/box_loss"], label="Val Box Loss", linestyle="--")
plt.plot(df["val/cls_loss"], label="Val Class Loss", linestyle="--")
plt.plot(df["val/dfl_loss"], label="Val DFL Loss", linestyle="--")

plt.title("YOLO Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"))
print("✔ Saved: loss_curve.png")


# -------------------------------------------------------
# 3. SYNTHETIC F1 CURVE USING PRECISION & RECALL
# -------------------------------------------------------
precision = df["metrics/precision(B)"]
recall = df["metrics/recall(B)"]

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

df_f1 = pd.DataFrame({
    "epoch": df["epoch"],
    "f1_score": f1_scores
})
df_f1.to_csv(os.path.join(OUT_DIR, "f1_scores.csv"), index=False)

plt.figure(figsize=(8,5))
plt.plot(df["epoch"], f1_scores, label="F1 Score", color="purple")
plt.title("Synthetic F1 Curve")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f1_curve.png"))
print("✔ Saved: f1_curve.png")


# -------------------------------------------------------
# 4. RMSE PER EPOCH USING mAP50(B)
# -------------------------------------------------------
mAP50 = df["metrics/mAP50(B)"]

rmse = np.sqrt(1 - mAP50)

df_rmse = pd.DataFrame({
    "epoch": df["epoch"],
    "rmse": rmse
})
df_rmse.to_csv(os.path.join(OUT_DIR, "rmse_per_epoch.csv"), index=False)

plt.figure(figsize=(8,5))
plt.plot(df["epoch"], rmse, label="RMSE", color="green")
plt.title("RMSE vs Epoch (Derived from mAP50)")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmse_curve.png"))
print("✔ Saved: rmse_curve.png")

print("\n🎉 All training logs exported successfully!")
