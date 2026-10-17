"""EDA: professional visualizations saved to reports/figures."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def main(data: str, figdir: str):
    df = pd.read_csv(data)
    fig_path = Path(figdir)
    fig_path.mkdir(parents=True, exist_ok=True)
    print(df.shape)
    print(df.describe().T)
    print(df.isna().sum())

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="target", hue="target", data=df, ax=ax, palette="Set2", legend=False)
    ax.set_title("Class balance (0=no disease, 1=disease)")
    fig.savefig(fig_path / "01_class_balance.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Histograms of numeric features
    num = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, c in enumerate(num):
        sns.histplot(df[c], kde=True, ax=axes[i])
        axes[i].set_title(c)
    axes[5].axis("off")
    fig.suptitle("Numeric feature distributions")
    fig.savefig(fig_path / "02_histograms.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.savefig(fig_path / "03_correlation.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Age vs cholesterol by target
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x="age", y="chol", hue="target", data=df, ax=ax, palette="Set1")
    ax.set_title("Age vs Cholesterol by target")
    fig.savefig(fig_path / "04_age_chol.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Chest pain type vs target
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="cp", hue="target", data=df, ax=ax, palette="Set2")
    ax.set_title("Chest pain type (cp) vs target")
    fig.savefig(fig_path / "05_cp_target.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures -> {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--figdir", default="reports/figures")
    args = ap.parse_args()
    main(args.data, args.figdir)
