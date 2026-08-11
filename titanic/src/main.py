"""Titanic サンプルデータの予測正答率を表示するスクリプト。

女性ルールと決定木を学習・比較し、決定木の構造を表示・描画する。

使い方:
    uv run python src/main.py [CSV_PATH]
    CSV_PATH 省略時は data/train.csv を使う
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from female_rule import FemaleRule
from tree import DecisionTree, FEATURES

REQUIRED_COLUMNS = {"Survived", "Sex"}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"エラー: 列が不足しています: {sorted(missing)}")
    if df["Survived"].isnull().any():
        sys.exit(
            "エラー: Survived に欠損があります (正答率の計算には正解ラベルが必要です)"
        )
    return df


def accuracy(model: object, df: pd.DataFrame) -> float:
    return float((model.predict(df) == df["Survived"]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Titanic サンプルデータの正答率を表示")
    parser.add_argument(
        "csv",
        nargs="?",
        default="data/train.csv",
        help="サンプルデータのパス (デフォルト: data/train.csv)",
    )
    args = parser.parse_args()

    df = load_data(Path(args.csv))
    train, val = df.iloc[:712], df.iloc[712:]  # 学習用 80% / 評価用 20%

    female = FemaleRule()
    tree = DecisionTree(max_depth=3).fit(train)

    print(f"データ: {args.csv} ({len(df)} 人) | 学習: {len(train)} 人 / 評価: {len(val)} 人")
    print(f"女性ルール    正答率: {accuracy(female, val):.4f}")
    print(f"決定木(検証)  正答率: {accuracy(tree, val):.4f}")
    print(f"決定木(学習)  正答率: {accuracy(tree, train):.4f}  <- max_depth を上げると過学習の目安になる")

    feature_names = ["Sex(女性=1)" if f == "Sex" else f for f in FEATURES]
    tree.print_rules(feature_names)
    # 図はフォント問題を避けるため ASCII ラベルのみ使う
    ascii_names = ["Sex(f=1)" if f == "Sex" else f for f in FEATURES]
    outdir = Path("figures")
    outdir.mkdir(exist_ok=True)
    tree.draw(ascii_names, outdir / "tree.png")


if __name__ == "__main__":
    main()
