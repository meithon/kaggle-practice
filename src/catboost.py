"""簡潔な CatBoost 風実装 (学習用)。

CatBoost の核となる 2 アイデアだけを抜き出す:

1. ordered target statistics — カテゴリ値を「ランダム順列で自分より前のサンプルの
   目的変数平均」で置換する。未来の情報を使わないのでリークしない (= シンプルな
   カテゴリ平均 encoding との差)。
2. gradient boosting — 浅い回帰木を残差 (= 損失の負勾配) に逐次フィットして
   学習率付きで足し合わせる。

本物は ordered boosting (サンプルごとに仮想モデル) なども行うが、学習目的なので省略。

使い方:
    uv run python src/catboost.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from predictor import Predictor
from tree import Node


class CatBoost(Predictor):
    # 特徴量構成。サブクラスで上書きして拡張できる
    NUMERIC = ["Age", "Fare"]
    CATEGORICAL = ["Sex", "Pclass"]

    def __init__(self, iterations: int = 100, learning_rate: float = 0.1,
                 depth: int = 3, min_samples_leaf: int = 5, seed: int = 0):
        self.iterations = iterations
        self.lr = learning_rate
        self.depth = depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees: list[Node] = []
        self.base = 0.0
        self.cat_ts: dict[str, dict[str, float]] = {}  # 列 -> カテゴリ -> TS値
        self.medians: pd.Series = pd.Series(dtype=float)

    # --- 1. ordered target statistics ---

    def _ordered_ts(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """カテゴリ値を「順列で自分より前の y 平均 + 平滑化」に置換する。"""
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(values))
        prior = y.mean()
        ts = np.empty(len(values))
        for p in range(len(perm)):
            i = perm[p]
            before = values[perm[:p]] == values[i]
            ts[i] = (y[perm[:p]][before].sum() + prior) / (before.sum() + 1)
        return ts

    @staticmethod
    def _features(df: pd.DataFrame) -> pd.DataFrame:
        """生 DataFrame から学習用特徴量を組み立てる (基準: 4 特徴量)。"""
        return df[["Age", "Fare", "Sex", "Pclass"]].copy()

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ列を学習済み TS 値へ、数値列を中央値補完で置き換える。"""
        X = self._features(df)
        for col in self.CATEGORICAL:
            X[col] = X[col].map(str).map(self.cat_ts[col]).fillna(self.base).astype(float)
        return X[self.NUMERIC + self.CATEGORICAL].fillna(self.medians)

    # --- 2. 回帰木 (残差用の浅い CART) ---

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        best = None
        for j in range(X.shape[1]):
            for thr in np.unique(X[:, j]):
                mask = X[:, j] <= thr
                if mask.all() or not mask.any():
                    continue
                sse = (np.var(y[mask]) * mask.sum()) + (np.var(y[~mask]) * (~mask).sum())
                if best is None or sse < best[0]:
                    best = (sse, j, float(thr))
        return None if best is None else (best[1], best[2])

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if depth >= self.depth or len(y) < self.min_samples_leaf or len(np.unique(y)) == 1:
            return Node(prob=y.mean(), n=len(y))
        split = self._best_split(X, y)
        if split is None:
            return Node(prob=y.mean(), n=len(y))
        j, thr = split
        mask = X[:, j] <= thr
        return Node(feature=j, threshold=thr,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1), n=len(y))

    def _tree_predict(self, X: np.ndarray, node: Node) -> np.ndarray:
        return np.array([self._row_predict(row, node) for row in X])

    @staticmethod
    def _row_predict(row: np.ndarray, node: Node) -> float:
        while node.prob is None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prob

    # --- 勾配ブースティング ---

    def fit(self, df: pd.DataFrame) -> "CatBoost":
        y = df["Survived"].to_numpy(dtype=float)
        self.base = y.mean()
        feat = self._features(df)
        self.medians = feat[self.NUMERIC].median(numeric_only=True)

        # 学習データでカテゴリ列ごとに ordered TS を計算し、カテゴリ->TS の写像を保存
        for col in self.CATEGORICAL:
            vals = feat[col].map(str).to_numpy()  # NaN も 'nan' 文字列として扱う
            ts = self._ordered_ts(vals, y)
            self.cat_ts[col] = {v: float(ts[vals == v].mean()) for v in np.unique(vals)}
        X = self._encode(df).to_numpy()

        # 反復ごとに残差 y - F へ回帰木をフィットして足す
        F = np.full(len(y), self.base)
        for _ in range(self.iterations):
            tree = self._build(X, y - F, 0)
            self.trees.append(tree)
            F += self.lr * self._tree_predict(X, tree)
        return self

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """生スコア F (0.5 以上で生存予測)。信頼度 = 0.5 からの距離。"""
        X = self._encode(df).to_numpy()
        return np.full(len(X), self.base) + self.lr * sum(
            self._tree_predict(X, t) for t in self.trees
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        score = self.predict_score(df)
        return pd.Series((score >= 0.5).astype(int), index=df.index)


def accuracy(model: Predictor, df: pd.DataFrame) -> float:
    return float((model.predict(df) == df["Survived"]).mean())


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    train, val = df.iloc[:712], df.iloc[712:]

    cb = CatBoost().fit(train)
    print(f"学習: {len(train)} 人 / 評価: {len(val)} 人")
    print(f"CatBoost(検証)  正答率: {accuracy(cb, val):.4f}")
    print(f"CatBoost(学習)  正答率: {accuracy(cb, train):.4f}")
    print(f"TS 写像 (Sex):   {cb.cat_ts['Sex']}")

    # 誤分類の内訳: 自信 (0.5 からの距離) が大きい順に外したケースを見る
    score = cb.predict_score(val)
    pred = (score >= 0.5).astype(int)
    err = val.copy()
    err["score"] = score.round(3)
    err["pred"] = pred
    wrong = err[err["pred"] != err["Survived"]].sort_values(
        "score", key=lambda s: (s - 0.5).abs(), ascending=False
    )
    print(f"\n誤分類: {len(wrong)} / {len(val)} 人 (上位10件を自信順に表示)")
    cols = ["Survived", "pred", "score", "Sex", "Pclass", "Age", "Fare"]
    print(wrong.head(10).to_string(index=True))


if __name__ == "__main__":
    main()
