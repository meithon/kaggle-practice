"""Kaggle 提出用の prediction を生成するスクリプト。

train.csv 全体でモデルを学習し、test.csv を予測して
submissions/submission.csv (PassengerId, Survived) を出力する。

使い方:
    uv run python src/make_submission.py [--model female|tree|rf] [--depth N]
"""

import argparse
from pathlib import Path

import pandas as pd

from female_rule import FemaleRule
from random_forest import RandomForest
from tree import DecisionTree


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle 提出ファイルを生成")
    parser.add_argument("--model", choices=["female", "tree", "rf"], default="tree")
    parser.add_argument("--depth", type=int, default=5,
                        help="モデルの深さ (tree: 決定木, rf: Random Forest)")
    parser.add_argument("--out", default="submissions/submission.csv",
                        help="出力ファイルパス")
    args = parser.parse_args()

    train = pd.read_csv(Path("data/train.csv"))
    test = pd.read_csv(Path("data/test.csv"))

    if args.model == "female":
        model = FemaleRule()
    elif args.model == "rf":
        # 単一モデル: 過学習曲線で最適だった浅めの複雑さ (depth 4 前後)
        model = RandomForest(n_estimators=500, max_depth=args.depth,
                             min_samples_leaf=1, seed=0).fit(train)
    else:
        model = DecisionTree(max_depth=args.depth).fit(train)

    surv = model.predict(test)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": surv})

    outdir = Path(args.out).parent
    outdir.mkdir(exist_ok=True)
    out = Path(args.out)
    sub.to_csv(out, index=False)
    print(f"出力: {out} ({len(sub)} 行)")


if __name__ == "__main__":
    main()
