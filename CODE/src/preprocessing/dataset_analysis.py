import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import config


def main():
    if not config.RAW_DATASET_DIR.exists():
        print(f"Dataset not found: {config.RAW_DATASET_DIR}"); return
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{"class_name": f.name, "image_count": len(list(f.glob("*")))}
                       for f in sorted(config.RAW_DATASET_DIR.iterdir()) if f.is_dir()])
    print(f"Classes: {len(df)}  Total: {df.image_count.sum()}  "
          f"Min: {df.image_count.min()}  Max: {df.image_count.max()}  Avg: {df.image_count.mean():.1f}")
    print(df.to_string())

    csv_path = config.REPORTS_DIR / "dataset_class_distribution.csv"
    df.to_csv(csv_path, index=False)

    fig_path = config.FIGURES_DIR / "dataset_class_distribution.png"
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x="class_name", y="image_count")
    plt.xticks(rotation=45, ha="right")
    plt.title("ArSignTutor Dataset Class Distribution")
    plt.xlabel("Class Name"); plt.ylabel("Number of Images")
    plt.tight_layout(); plt.savefig(fig_path, dpi=300); plt.close()

    print(f"Saved: {csv_path}  |  {fig_path}")


if __name__ == "__main__":
    main()
