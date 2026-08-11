"""ドメイン特徴量を「全体の forest」に入れた場合の CV 比較。

rf_domain.py は特徴量を全データの forest に足して「領域」で測ったが、
ここでは「全体として forest が良くなるか」を stratified k-fold CV で比較する。

使い方:
    uv run python src/rf_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from random_forest import RandomForest, FEATURES
from rf_cv import stratified_kfold

CANDIDATES = {
    "FamilySize": lambda d: d["SibSp"] + d["Parch"] + 1,
    "IsAlone": lambda d: (d["SibSp"] + d["Parch"] == 0).astype(int),
    "HasCabin": lambda d: d["Cabin"].notna().astype(int),
}


def cv_acc(df: pd.DataFrame, feats: list[str], k: int = 5) -> np.ndarray:
    """stratified k-fold CV で各 fold の正答率を返す。"""
    y = df["Survived"].to_numpy()
    accs = []
    for val_idx in stratified_kfold(len(df), y, k, seed=0):
        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
        rf = RandomForest(n_estimators=500, max_depth=8, min_samples_leaf=1,
                          seed=0, features=feats).fit(df.iloc[tr_idx])
        pred = rf.predict(df.iloc[val_idx]).to_numpy()
        accs.append(float((pred == y[val_idx]).mean()))
    return np.array(accs)


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    d = df.copy()
    for name, fn in CANDIDATES.items():
        d[name] = fn(d)

    sets = [("ベース(FEATURES)", FEATURES)]
    for name in CANDIDATES:
        sets.append((f"+{name}", FEATURES + [name]))
    sets.append(("+全部", FEATURES + list(CANDIDATES)))

    print("== ドメイン特徴量を全体の forest に入れた場合の CV 比較 (k=5) ==")
    print(f"{'特徴量':<20}{'CV正答率 (mean ± std)':>22}")
    for label, feats in sets:
        accs = cv_acc(d, feats)
        print(f"{label:<20}{np.mean(accs):.4f} ± {np.std(accs):.4f}")


if __name__ == "__main__":
    main()
