"""CatBoost の新情報版: 同乗グループ (Ticket) と 1人あたり運賃 を足す。

catboost_all.py (全カラム - ユニーク列) を継承し、モデル入力に無い新情報を 2 つ追加:

- TicketGroup: 同じ Ticket 番号を共有する乗客数 = 一緒に乗船したグループ。
  SibSp/Parch (家族) では拾えない「同乗者」を捉える。学習データから写像を計算。
- FarePerPerson: Fare / TicketGroup。運賃はチケット単位なので、人数で割って
  1人あたりの支払いに直す。除算は木の単列閾値分割では表現しにくい新表現力。

使い方:
    uv run python src/catboost_ticket.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catboost_all import CatBoostAll, plot_misclassifications


class CatBoostTicket(CatBoostAll):
    NUMERIC = ["Age", "SibSp", "Parch", "Fare", "TicketGroup", "FarePerPerson"]
    CATEGORICAL = ["Pclass", "Sex", "Cabin", "Embarked"]

    def fit(self, df: pd.DataFrame) -> "CatBoostTicket":
        # 新情報: 同チケット(同乗グループ)人数。学習データでの写像を保存
        self._ticket_size = df["Ticket"].value_counts()
        return super().fit(df)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = CatBoostAll._features(df)
        size = df["Ticket"].map(self._ticket_size).fillna(1.0)  # 未知チケットは 1 人
        X["TicketGroup"] = size
        X["FarePerPerson"] = df["Fare"] / size
        return X


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    train, val = df.iloc[:712], df.iloc[712:]

    for name, model in [("catboost_all (旧)", CatBoostAll), ("+ 新情報 (新)", CatBoostTicket)]:
        cb = model().fit(train)
        p = (cb.predict_score(val) >= 0.5).astype(int)
        pt = (cb.predict(train) == train["Survived"]).mean()
        print(f"{name}: 検証 {float((p == val['Survived']).mean()):.4f} / 学習 {float(pt):.4f}")

    cb = CatBoostTicket().fit(train)
    score = cb.predict_score(val)
    pred = (score >= 0.5).astype(int)
    print(f"\n新情報の TS 写像 (Sex): {cb.cat_ts['Sex']}")

    # 誤分類テーブル
    err = val.copy()
    err["CabinDeck"] = err["Cabin"].str[0].fillna("None")
    err["TicketGroup"] = err["Ticket"].map(cb._ticket_size).fillna(1.0)
    err["FarePerPerson"] = err["Fare"] / err["TicketGroup"]
    err["score"] = score.round(3)
    err["pred"] = pred
    wrong = err[err["pred"] != err["Survived"]].sort_values(
        "score", key=lambda s: (s - 0.5).abs(), ascending=False
    )
    print(f"\n誤分類: {len(wrong)} / {len(val)} 人 (上位10件を自信順に表示)")
    cols = ["Survived", "pred", "score", "Sex", "Pclass", "Age", "TicketGroup", "FarePerPerson", "CabinDeck"]
    print(wrong[cols].head(10).to_string(index=False))

    outdir = Path("figures")
    outdir.mkdir(exist_ok=True)
    plot_misclassifications(val, score, pred, outdir / "misclassification_ticket.png")


if __name__ == "__main__":
    main()
