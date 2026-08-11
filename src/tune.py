"""ハイパーパラメータ (depth, iterations) のグリッド探索 + 最良で全モデル比較。

使い方:
    uv run python src/tune.py
"""

from pathlib import Path

import pandas as pd

from cv import cv_accuracy
from catboost import CatBoost
from catboost_family import CatBoostFamily
from catboost_all import CatBoostAll
from catboost_ticket import CatBoostTicket

MODELS = [
    ("baseline (4特徴)", CatBoost),
    ("+FamilySize/Title", CatBoostFamily),
    ("全カラム-ユニーク", CatBoostAll),
    ("+TicketGroup/FPP", CatBoostTicket),
]


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))

    # グリッド探索は最速の baseline で行い、最良パラメータを全モデルに適用
    print("グリッド探索 (baseline, lr=0.1)")
    best = None
    for depth in (2, 3, 4, 5):
        for iters in (50, 100, 200):
            accs = cv_accuracy(CatBoost, df, depth=depth, iterations=iters)
            mean = float(accs.mean())
            flag = ""
            if best is None or mean > best[0]:
                best = (mean, depth, iters)
                flag = " <- best"
            print(f"depth={depth} iterations={iters}: {mean:.4f} ± {accs.std():.4f}{flag}")

    mean, depth, iters = best
    print(f"\n最良パラメータ: depth={depth}, iterations={iters} ({mean:.4f})")

    print("\n最良パラメータで全モデル比較 (5-fold CV)")
    for name, cls in MODELS:
        a = cv_accuracy(cls, df, depth=depth, iterations=iters)
        print(f"{name:20s}: {a.mean():.4f} ± {a.std():.4f}")


if __name__ == "__main__":
    main()
