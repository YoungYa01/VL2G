from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("/mnt/d/research_vl2g/final_tables/student_distillation_summary.csv")
out_dir = Path("/mnt/d/research_vl2g/final_figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# 只看 LISA 和 GT 的 100/500 趋势
trend = df[df["method"].isin(["Student-LISA", "Student-GT"])].copy()
trend = trend.sort_values(["method", "num_train"])

plt.figure(figsize=(7.5, 5))

for method in ["Student-LISA", "Student-GT"]:
    sub = trend[trend["method"] == method]
    plt.plot(sub["num_train"], sub["iou"], marker="o", linewidth=2, label=f"{method} IoU")
    for _, row in sub.iterrows():
        plt.text(row["num_train"], row["iou"] + 0.01, f"{row['iou']:.3f}", ha="center")

plt.xlabel("Number of Training Images")
plt.ylabel("IoU")
plt.title("Data Scale Trend: Student-LISA vs Student-GT")
plt.ylim(0.60, 0.80)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out_path = out_dir / "scale_trend_lisa_vs_gt.png"
plt.savefig(out_path, dpi=220)
print("saved:", out_path)
