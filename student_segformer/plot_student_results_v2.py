from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = Path("/mnt/d/research_vl2g/final_tables/student_distillation_summary.csv")
out_dir = Path("/mnt/d/research_vl2g/final_figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

df["display_name"] = df["method"] + "-" + df["num_train"].astype(str)

metrics = ["iou", "dice", "precision", "recall"]
metric_names = ["IoU", "Dice", "Precision", "Recall"]

x = np.arange(len(df))
width = 0.18

plt.figure(figsize=(13, 5.8))

for i, metric in enumerate(metrics):
    plt.bar(x + (i - 1.5) * width, df[metric], width, label=metric_names[i])

plt.xticks(x, df["display_name"], rotation=25, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Student Model Comparison under Different Training Labels and Data Scales")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out_path = out_dir / "student_model_comparison_100_500.png"
plt.savefig(out_path, dpi=220)
print("saved:", out_path)
