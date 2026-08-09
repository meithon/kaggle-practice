"""Titanic データの可視化スクリプト。

使い方:
    uv run python src/visualize.py

figures/ に PNG を出力する。
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- データ読込 ---
df = pd.read_csv("data/train.csv")
print("件数:", len(df), "| 列:", list(df.columns))

sns.set_theme(style="whitegrid")
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

# 1. 性別ごとの生存率
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df, x="Sex", y="Survived", ax=ax)
ax.set_title("Survival rate by Sex")
fig.savefig(OUTDIR / "survival_by_sex.png", bbox_inches="tight")
plt.close(fig)

# 2. 客室クラスごとの生存率
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df, x="Pclass", y="Survived", ax=ax)
ax.set_title("Survival rate by Pclass")
fig.savefig(OUTDIR / "survival_by_pclass.png", bbox_inches="tight")
plt.close(fig)

# 3. 年齢分布（生存/死亡別）
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, ax=ax)
ax.set_title("Age distribution by Survival")
fig.savefig(OUTDIR / "age_by_survival.png", bbox_inches="tight")
plt.close(fig)

print("出力:", sorted(p.name for p in OUTDIR.iterdir()))
