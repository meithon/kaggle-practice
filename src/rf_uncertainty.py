"""ノイズ (不確実性) の多い領域の特定。

deep/shallow の比較 (u = 誤差差) はノイズを学習するので止めた。代わりに
1 つのモデル (RandomForest) 自身の不確実性を使う:

- oob_std = 各乗客について「OOB の木ごとの予測」のばらつき。
  木がバラつくほど答えが決まらない = その領域のデータはノイズが多い。

検証: oob_std が大きいほど実際の OOB 誤答率も上がるか (5 分位で確認)
特定: oob_std を浅い回帰木で分割し、ノイズの多い/少ない領域を出す

使い方:
    uv run python src/rf_uncertainty.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from random_forest import (RandomForest, RegressionTree, features, prep,
                           FEATURES, leaf_regions)


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    y = df["Survived"].to_numpy()
    rf = RandomForest(n_estimators=500, max_depth=8, min_samples_leaf=1,
                      seed=0).fit(df)
    std = rf.oob_std
    err = (rf.oob_scores >= .5) != y  # OOB 誤り

    # 1. 不確実性 (oob_std) が実際の誤りを予測できているか
    print("== oob_std の 5 分位と OOB 誤答率 ==")
    labels = ["1(低)", "2", "3", "4", "5(高)"]
    q = pd.qcut(std, 5, labels=labels)
    print(f"{'分位':<8}{'人数':>5}{'std平均':>8}{'OOB誤答率':>10}")
    for lab in labels:
        m = q == lab
        print(f"{lab:<8}{m.sum():>5}{std[m].mean():>8.3f}{err[m].mean():>10.3f}")

    # 2. oob_std を回帰木で分割 → ノイズの多い/少ない領域
    print("\n== ノイズの多い領域 (oob_std を回帰木で分割) ==")
    med = features(df).median(numeric_only=True)
    X = prep(df, med).to_numpy()
    g = RegressionTree(max_depth=3, min_samples_leaf=20, seed=0).fit(X, std)
    print(f"{'領域':<36}{'n':>5}{'std平均':>8}{'誤答率':>8}")
    for rule, rows in leaf_regions(g.root, X, FEATURES):
        print(f"{rule:<36}{len(rows):>5}{std[rows].mean():>8.3f}{err[rows].mean():>8.3f}")

    # 3. 参考: 成人低運賃は本当に不確実性が高いか
    adult_lo = (X[:, 2] > 11) & (X[:, 3] <= 13.9)
    print(f"\n参考: 成人低運賃 (Age>11, Fare<=13.9) {adult_lo.sum()} 人"
          f" — std 平均 {std[adult_lo].mean():.3f} (全体 {std.mean():.3f}),"
          f" 誤答率 {err[adult_lo].mean():.3f} (全体 {err.mean():.3f})")


if __name__ == "__main__":
    main()
