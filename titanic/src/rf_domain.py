"""ノイズの多い領域 → ドメイン知識の仮説 → データで検証、というフロー。

分析は「どこがノイズか」を教えるが、「なぜ / どうすれば良いか」という
ドメイン知識はデータから自動では出ない。それは人間/LLM の仮説。
このスクリプトは「その仮説 (特徴量) を加えると領域の誤答率が下がるか」を
OOB で検証する (仮説が本当に有効かを測る)。

フロー:
1. oob_std でノイズの多い領域を特定 (回帰木の葉のうち誤答率最大)
2. その領域に効きそうなドメイン特徴量を仮説として挙げる
3. 各候補を加えて再学習 → 領域の OOB 誤答率が下がるか比較

使い方:
    uv run python src/rf_domain.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from random_forest import (RandomForest, RegressionTree, features, prep,
                           FEATURES, leaf_regions)

# ドメイン知識の仮説 (数値特徴量にできるもの)。Titanic の実データに基づく。
CANDIDATES = {
    "FamilySize": lambda d: d["SibSp"] + d["Parch"] + 1,   # 家族構成 (救命ボート行動)
    "IsAlone": lambda d: (d["SibSp"] + d["Parch"] == 0).astype(int),
    "HasCabin": lambda d: d["Cabin"].notna().astype(int),  # デッキ/救命ボートへのアクセス
}


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    y = df["Survived"].to_numpy()

    # --- ベースライン (FEATURES のみ) ---
    rf0 = RandomForest(n_estimators=500, max_depth=8, min_samples_leaf=1,
                       seed=0).fit(df)
    std = rf0.oob_std
    err0 = (rf0.oob_scores >= .5) != y

    # 1. ノイズの多い領域を特定: oob_std を回帰木で分割し、誤答率最大の葉を選ぶ
    med = features(df).median(numeric_only=True)
    X = prep(df, med).to_numpy()
    g = RegressionTree(max_depth=3, min_samples_leaf=20, seed=0).fit(X, std)
    cand = [r for r in leaf_regions(g.root, X, FEATURES) if len(r[1]) >= 50]
    rule, rows = max(cand, key=lambda r: err0[r[1]].mean())
    print("== ノイズの多い領域 (oob_std 回帰木の葉で誤答率最大) ==")
    print(f"領域: {rule}  (n={len(rows)}, OOB誤答率 {err0[rows].mean():.3f})")

    # 2 + 3. ドメイン仮説を検証: 特徴量を足して再学習 → 領域の誤答率が下がるか
    print("\n== ドメイン知識の仮説を検証 (特徴量を足すと領域の誤答率が下がるか) ==")
    print(f"{'仮説(特徴量)':<14}{'領域誤答率':>10}{'領域std':>9}{'全体誤答率':>10}")
    print(f"{'ベース':<14}{err0[rows].mean():>10.3f}{std[rows].mean():>9.3f}"
          f"{err0.mean():>10.3f}")
    for name, fn in CANDIDATES.items():
        d = df.copy()
        d[name] = fn(d)
        rf = RandomForest(n_estimators=500, max_depth=8, min_samples_leaf=1,
                          seed=0, features=FEATURES + [name]).fit(d)
        err = (rf.oob_scores >= .5) != y
        print(f"{name:<14}{err[rows].mean():>10.3f}{rf.oob_std[rows].mean():>9.3f}"
              f"{err.mean():>10.3f}")


if __name__ == "__main__":
    main()
