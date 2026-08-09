"""Titanic サンプルデータの予測正答率を表示するスクリプト。

サンプルデータ (Survived を含む CSV) を受け取り、
「女性なら生存、男性なら死亡」のルールで予測して正答率を表示する。

使い方:
    uv run python src/main.py [CSV_PATH]
    CSV_PATH 省略時は data/train.csv を使う
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

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


def predict(df: pd.DataFrame) -> pd.Series:
    """女性なら生存 (1)、男性なら死亡 (0) と予測する。"""
    # 決定木
    return (df["Sex"].str.lower() == "female").astype(int)


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
    pred = predict(df)
    acc = (pred == df["Survived"]).mean()

    print(f"データ: {args.csv} ({len(df)} 人)")
    print("予測ルール: 女性=生存, 男性=死亡")
    print(f"正答率: {acc:.4f}")


if __name__ == "__main__":
    main()
