"""スクラッチ実装の CART 決定木 (数値特徴量のみ、ジニ不純度で分割)。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import Predictor

FEATURES = ["Pclass", "Sex", "Age", "Fare"]  # Sex は fit 内で 0/1 に変換


@dataclass
class Node:
    feature: int | None = None   # 分岐に使う特徴量の列番号
    threshold: float | None = None  # 分岐しきい値 (X[:, feature] <= threshold)
    left: "Node | None" = None
    right: "Node | None" = None
    prob: float | None = None    # 葉のみ: 生存確率
    n: int = 0                   # このノードに到達したサンプル数


class DecisionTree(Predictor):
    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root: Node | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p**2 - (1.0 - p) ** 2

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
        """重み付きジニ不純度が最小になる (列, しきい値) を探す。"""
        best: tuple[float, int, float] | None = None
        for j in range(X.shape[1]):
            for thr in np.unique(X[:, j]):
                mask = X[:, j] <= thr
                if mask.all() or not mask.any():
                    continue
                weighted = (
                    len(y[mask]) * self._gini(y[mask])
                    + len(y[~mask]) * self._gini(y[~mask])
                ) / len(y)
                if best is None or weighted < best[0]:
                    best = (weighted, j, float(thr))
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # 停止条件: 深さ上限・サンプル数下限・ラベルが単一
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_leaf
            or len(np.unique(y)) == 1
        ):
            return Node(prob=y.mean(), n=len(y))
        split = self._best_split(X, y)
        if split is None:
            return Node(prob=y.mean(), n=len(y))
        _, j, thr = split
        mask = X[:, j] <= thr
        return Node(
            feature=j,
            threshold=thr,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
            n=len(y),
        )

    def _predict_row(self, row: np.ndarray, node: Node) -> float:
        while node.prob is None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prob

    # --- 可視化 ---

    def print_rules(self, feature_names: list[str]) -> None:
        """インデントで木構造をテキスト表示する。"""
        def walk(node: Node, depth: int) -> None:
            pad = "  " * depth
            if node.prob is not None:
                print(f"{pad}→ 生存 {node.prob:.2f} (n={node.n})")
                return
            fname = feature_names[node.feature]
            thr = node.threshold
            print(f"{pad}├─ {fname} <= {thr:.1f}? (n={node.n})")
            walk(node.left, depth + 1)
            print(f"{pad}└─ {fname} >  {thr:.1f}")
            walk(node.right, depth + 1)

        walk(self.root, 0)

    def draw(self, feature_names: list[str], out: Path) -> None:
        """matplotlib で木を描いて PNG に保存する。"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        pos: dict[int, tuple[float, float]] = {}  # id(node) -> (x, y)

        def layout(node: Node, depth: int, xmin: float, xmax: float) -> int:
            """サブツリーに x 区間を与え、葉の数で x 中心を割り当てる。"""
            if node.prob is not None:
                pos[id(node)] = ((xmin + xmax) / 2, -depth)
                return 1
            nl = layout(node.left, depth + 1, xmin, xmin + (xmax - xmin) / 2)
            nr = layout(node.right, depth + 1, xmin + (xmax - xmin) / 2, xmax)
            pos[id(node)] = ((pos[id(node.left)][0] + pos[id(node.right)][0]) / 2, -depth)
            return nl + nr

        layout(self.root, 0, 0, 1)

        def draw_node(node: Node) -> None:
            x, y = pos[id(node)]
            if node.prob is not None:
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.07, y - 0.18), 0.14, 0.36,
                        facecolor="#4C72B0", alpha=0.5, edgecolor="black",
                    )
                )
                ax.text(x, y, f"surv {node.prob:.2f}\nn={node.n}",
                        ha="center", va="center", fontsize=9)
            else:
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.13, y - 0.18), 0.26, 0.36,
                        facecolor="#DD8452", alpha=0.5, edgecolor="black",
                    )
                )
                fname = feature_names[node.feature]
                ax.text(x, y, f"{fname} <= {node.threshold:.1f}\nn={node.n}",
                        ha="center", va="center", fontsize=9)
                cx, cy = pos[id(node.left)]
                ax.plot([x, cx], [y, cy], color="black", lw=1)
                cx, cy = pos[id(node.right)]
                ax.plot([x, cx], [y, cy], color="black", lw=1)
                draw_node(node.left)
                draw_node(node.right)

        draw_node(self.root)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"出力: {out}")

    def fit(self, df: pd.DataFrame) -> "DecisionTree":
        X = df[FEATURES].copy()
        X["Sex"] = (X["Sex"].str.lower() == "female").astype(int)
        X = X.fillna(X.median(numeric_only=True))  # Age の欠損を中央値で補完
        self.root = self._build(X.to_numpy(), df["Survived"].to_numpy(), 0)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[FEATURES].copy()
        X["Sex"] = (X["Sex"].str.lower() == "female").astype(int)
        X = X.fillna(X.median(numeric_only=True))
        prob = [self._predict_row(row, self.root) for row in X.to_numpy()]
        return pd.Series((np.array(prob) >= 0.5).astype(int), index=df.index)
