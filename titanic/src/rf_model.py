"""単一の Random Forest — CV で複雑さ (深さ) を選ぶ (シンプル版)。

deep/shallow の使い分けをやめ、1 つのモデル (RandomForest) を 1 本のノブ
(max_depth) で最適化する。

責務ごとの構成 (各関数は 1 つの責務だけを持つ):
- make_rf()     : モデル生成 (ハイパラの定義場所はここだけ)
- cv_accuracy() : k-fold CV で「各 fold の正答率」を返す (評価だけ)
- sweep_depths(): 候補深さを CV で比較し、集計表を返す (比較だけ)
- select_best() : 表から CV 平均最大の行を選ぶ (選択だけ)
- main()        : それらを組み立て、表示・保存する (構成だけ)

使い方:
    uv run python src/rf_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from random_forest import RandomForest
from rf_cv import stratified_kfold


def make_rf(max_depth: int | None) -> RandomForest:
    """この実験で使う Random Forest を生成する。"""
    return RandomForest(n_estimators=500, max_depth=max_depth,
                        min_samples_leaf=1, seed=0)


def cv_accuracy(df: pd.DataFrame, max_depth: int | None, k: int = 5,
                seed: int = 0) -> np.ndarray:
    """stratified k-fold CV で各 fold の正答率 (長さ k の配列) を返す。"""
    y = df["Survived"].to_numpy()
    accs = []
    for val_idx in stratified_kfold(len(df), y, k, seed):
        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
        rf = make_rf(max_depth).fit(df.iloc[tr_idx])
        pred = rf.predict(df.iloc[val_idx]).to_numpy()
        accs.append(float((pred == y[val_idx]).mean()))
    return np.array(accs)


def sweep_depths(df: pd.DataFrame, depths: list[int | None],
                 k: int = 5) -> pd.DataFrame:
    """候補深さごとに CV 正答率を集計した表を返す。"""
    rows = []
    for depth in depths:
        accs = cv_accuracy(df, depth, k)
        rows.append({
            "max_depth": str(depth),
            "cv_mean": float(accs.mean()),
            "cv_std": float(accs.std()),
            "cv_min": float(accs.min()),
            "cv_max": float(accs.max()),
        })
    return pd.DataFrame(rows)


def select_best(table: pd.DataFrame) -> pd.Series:
    """CV 平均が最大の行 (最良の深さ) を返す。"""
    return table.loc[table.cv_mean.idxmax()]


def parse_depth(s: str) -> int | None:
    """表の max_depth 文字列 ("4" / "None") を値に戻す。"""
    return None if s == "None" else int(s)


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    y = df["Survived"].to_numpy()

    k = 5
    depths = [1, 2, 3, 4, 6, 8, 10, None]
    print(f"== 単一の Random Forest — stratified k={k} CV で深さを選ぶ ==")
    print("(n_estimators=500, min_samples_leaf=1)")

    # 1. 候補深さを CV で比較
    table = sweep_depths(df, depths, k)
    print(table.to_string(index=False, float_format="%.4f"))

    # 2. 最良の深さを選ぶ
    best = select_best(table)
    print(f"\n→ 最良: max_depth={best.max_depth} "
          f"(CV {best.cv_mean:.4f} ± {best.cv_std:.4f})")

    # 3. 全データで最終モデルを学習し、OOB で確認
    final = make_rf(parse_depth(best.max_depth)).fit(df)
    print(f"最終モデル: RandomForest(500, max_depth={best.max_depth}) "
          f"を全 {len(df)} 人で学習")
    print(f"  OOB 正答率: {((final.oob_scores >= .5) == y).mean():.4f}")

    # 4. 表を保存 (nb から読む用)
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    out = outdir / "rf_cv_depth.csv"
    table.to_csv(out, index=False)
    print(f"出力: {out}")


if __name__ == "__main__":
    main()
