"""5-fold 交差検証で CatBoost の全バリアントを平均精度で比較する。

使い方:
    uv run python src/cv.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoost
from catboost_family import CatBoostFamily
from catboost_all import CatBoostAll
from catboost_ticket import CatBoostTicket

MODELS = {
    "baseline (4特徴)": CatBoost,
    "+FamilySize/Title": CatBoostFamily,
    "全カラム-ユニーク": CatBoostAll,
    "+TicketGroup/FPP": CatBoostTicket,
}


def cv_accuracy(model_cls, df, n_splits=5, seed=0, **model_kwargs):
    """Stratified 5-fold CV で各 fold の正答率を返す。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["Survived"].to_numpy()
    accs = []
    for tr_idx, va_idx in skf.split(df, y):
        model = model_cls(**model_kwargs).fit(df.iloc[tr_idx])
        score = model.predict_score(df.iloc[va_idx])
        accs.append(((score >= 0.5).astype(int) == y[va_idx]).mean())
    return np.array(accs)


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    print(f"5-fold CV (stratified, seed=0) / データ {len(df)} 行\n")
    results = []
    for name, cls in MODELS.items():
        accs = cv_accuracy(cls, df)
        results.append((name, accs))
        print(f"{name:20s}: {accs.mean():.4f} ± {accs.std():.4f}  folds={np.round(accs, 3)}")

    best = max(results, key=lambda r: r[1].mean())
    print(f"\n最良: {best[0]} ({best[1].mean():.4f} ± {best[1].std():.4f})")


if __name__ == "__main__":
    main()
