"""CatBoost の全特徴量版: 完全ユニーク列を除いた全カラムを使う。

CatBoost (src/catboost.py) を継承し、NUMERIC/CATEGORICAL と _features だけを
上書きする。Cabin は頭文字 = デッキ (欠損は "None" カテゴリ)。

除外した列: PassengerId / Name / Ticket (ほぼ全行ユニーク = 丸暗記の材料で
一般化しない。カテゴリ TS の平滑化で害は軽減されるが、情報としても無い)。

使い方:
    uv run python src/catboost_all.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catboost import CatBoost


def plot_misclassifications(val: pd.DataFrame, score: np.ndarray,
                            pred: pd.Series, out: Path) -> None:
    """誤分類を 2 パネルで描く: (左) 予測スコア分布 (右) 混同行列。"""
    actual = val["Survived"].to_numpy()
    correct = pred == actual

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左: スコア strip plot (y = 実際のクラス, 色 = 正誤) ---
    ax = axes[0]
    rng = np.random.default_rng(1)
    xs = score + rng.uniform(-0.015, 0.015, len(score))
    ys = actual.astype(float) + rng.uniform(-0.3, 0.3, len(score))
    ax.scatter(xs[correct], ys[correct], s=18, c="#2ca02c", alpha=0.55, label="correct")
    ax.scatter(xs[~correct], ys[~correct], s=48, c="#d62728", marker="X",
               edgecolor="black", label="misclassified")
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.text(0.03, 1.28, "FN: survived, predicted dead", fontsize=8, color="#d62728")
    ax.text(0.53, -0.52, "FP: died, predicted survived", fontsize=8, color="#d62728")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Died (actual 0)", "Survived (actual 1)"])
    ax.set_xlabel("Prediction score F")
    ax.set_ylim(-0.65, 1.65)
    ax.set_title("Misclassified by score and actual class")
    ax.legend(loc="upper left")

    # --- 右: 混同行列 ---
    ax = axes[1]
    cm = np.zeros((2, 2), dtype=int)
    for a in (0, 1):
        for p in (0, 1):
            cm[a, p] = ((actual == a) & (pred == p)).sum()
    im = ax.imshow(cm, cmap="Blues")
    for a in (0, 1):
        for p in (0, 1):
            white = cm[a, p] > cm.max() / 2
            ax.text(p, a, cm[a, p], ha="center", va="center", fontsize=15,
                    color="white" if white else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["predicted died", "predicted survived"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual died", "actual survived"])
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"出力: {out}")


class CatBoostAll(CatBoost):
    # 完全ユニーク列 (PassengerId, Name, Ticket) は除外
    NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
    CATEGORICAL = ["Pclass", "Sex", "Cabin", "Embarked"]

    @staticmethod
    def _features(df: pd.DataFrame) -> pd.DataFrame:
        X = df[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Embarked"]].copy()
        X["Cabin"] = df["Cabin"].str[0].fillna("None")  # 頭文字 = デッキ (A-G)
        return X


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    train, val = df.iloc[:712], df.iloc[712:]

    cb = CatBoostAll().fit(train)
    score = cb.predict_score(val)
    pred = (score >= 0.5).astype(int)
    print(f"学習: {len(train)} 人 / 評価: {len(val)} 人")
    print(f"CatBoostAll(検証)  正答率: {float((pred == val['Survived']).mean()):.4f}")
    print(f"CatBoostAll(学習)  正答率: {float((cb.predict(train) == train['Survived']).mean()):.4f}")

    # 外れ値フラグ (IQR 法: Fare > Q3 + 1.5*IQR)。閾値は train で決め val に適用
    q1, q3 = train["Fare"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    val_outlier = val["Fare"] > upper
    acc = lambda s: float((pred[s] == val.loc[s, "Survived"]).mean())
    print(f"Fare 外れ値閾値: > {upper:.1f} (Q1={q1:.1f}, Q3={q3:.1f})")
    print(f"外れ値行 {val_outlier.sum()} 人 正答率: {acc(val_outlier):.4f}")
    print(f"通常行   {(~val_outlier).sum()} 人 正答率: {acc(~val_outlier):.4f}")

    # 誤分類の内訳: 自信 (0.5 からの距離) が大きい順に外したケースを見る
    err = val.copy()
    err["CabinDeck"] = err["Cabin"].str[0].fillna("None")
    err["FamilySize"] = err["SibSp"] + err["Parch"] + 1
    err["Outlier"] = val_outlier
    err["score"] = score.round(3)
    err["pred"] = pred
    wrong = err[err["pred"] != err["Survived"]].sort_values(
        "score", key=lambda s: (s - 0.5).abs(), ascending=False
    )
    print(f"\n誤分類: {len(wrong)} / {len(val)} 人 (上位10件を自信順に表示)")
    cols = ["Survived", "pred", "score", "Outlier", "Sex", "Pclass", "Age", "SibSp", "Parch", "FamilySize", "Fare", "CabinDeck", "Embarked"]
    print(wrong[cols].head(10).to_string(index=False))

    outdir = Path("figures")
    outdir.mkdir(exist_ok=True)
    plot_misclassifications(val, score, pred, outdir / "misclassification.png")


if __name__ == "__main__":
    main()
