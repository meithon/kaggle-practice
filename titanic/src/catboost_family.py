"""CatBoost の拡張版: FamilySize と Title を加えた特徴量構成。

CatBoost (src/catboost.py) を継承し、NUMERIC/CATEGORICAL と _features だけを
上書きして特徴量を増やす。基底クラスを変えずに試せる。

使い方:
    uv run python src/catboost_family.py
"""

from pathlib import Path

import pandas as pd

from catboost import CatBoost


class CatBoostFamily(CatBoost):
    NUMERIC = ["Age", "Fare", "FamilySize"]
    CATEGORICAL = ["Sex", "Pclass", "Title"]

    @staticmethod
    def _features(df: pd.DataFrame) -> pd.DataFrame:
        X = df[["Age", "Fare", "Sex", "Pclass"]].copy()
        X["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # 本人含む家族人数
        X["Title"] = df["Name"].str.extract(r",\s*([^.\s]+)\.")[0].fillna("Other")
        return X


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    train, val = df.iloc[:712], df.iloc[712:]

    cb = CatBoostFamily().fit(train)
    score = cb.predict_score(val)
    pred = (score >= 0.5).astype(int)
    print(f"学習: {len(train)} 人 / 評価: {len(val)} 人")
    print(f"CatBoostFamily(検証)  正答率: {float((pred == val['Survived']).mean()):.4f}")
    print(f"CatBoostFamily(学習)  正答率: {float((cb.predict(train) == train['Survived']).mean()):.4f}")

    # 誤分類の内訳: 自信 (0.5 からの距離) が大きい順に外したケースを見る
    err = val.copy()
    err["FamilySize"] = err["SibSp"] + err["Parch"] + 1
    err["Title"] = err["Name"].str.extract(r",\s*([^.\s]+)\.")[0].fillna("Other")
    err["score"] = score.round(3)
    err["pred"] = pred
    wrong = err[err["pred"] != err["Survived"]].sort_values(
        "score", key=lambda s: (s - 0.5).abs(), ascending=False
    )
    print(f"\n誤分類: {len(wrong)} / {len(val)} 人 (上位10件を自信順に表示)")
    cols = ["Survived", "pred", "score", "Sex", "Pclass", "Age", "FamilySize", "Title"]
    print(wrong.head(10).to_string(index=True))


if __name__ == "__main__":
    main()
