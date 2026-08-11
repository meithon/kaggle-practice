"""スクラッチ Random Forest + 「複雑さが効く領域」の学習 (diff.md の実装)。

diff.md の提案を 3 ステップで実行する:

1. 1本の決定木 → Random Forest で何が改善するか
   - 1本の深い木は学習データを丸暗記する (過学習)。Random Forest は
     ブートストラップ標本ごとに大量の深い木を作り、平均することで
     各木が拾った偶然のノイズを相殺する (= 分散の削減)。
2. deep RF vs shallow RF の OOB (out-of-bag) 予測で局所比較をする
   - 各木は学習に使わなかった行 (OOB) に対して予測できる → CV を回さなくても
     「未知データとして扱ったときの予測」が得られる。
   - M_c(x) - M_s(x): 細かい局所構造まで使ったことで予測がどれだけ変わったか
   - e_c(x) = (M_c^OOB(x) - y)^2, e_s(x) = (M_s^OOB(x) - y)^2
   - u(x) = e_s(x) - e_c(x): u > 0 ⇔ 複雑さ (深さ) が効いた領域
3. g(x) → u を浅い回帰木で学習し、「複雑さが効く領域」をモデル化する

設計:
- 木は tree.py と同じ CART (gini 分割、Sex→0/1、Age/Fare 中央値補完)。
- 分割探索は累積和でベクトル化 (tree.py のナイーブ版より速い)。
- min_samples_leaf は sklearn と同じく「分割後の両葉が満たすべき最小サンプル数」。
- sklearn の max_features (分割ごとの特徴量サブセット) は省略: 純粋なバギング。

使い方:
    uv run python src/random_forest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from predictor import Predictor
from tree import FEATURES, DecisionTree, Node


def features(df: pd.DataFrame) -> pd.DataFrame:
    """生 DataFrame から学習用特徴量 (基準 4 つ) を作る。Sex は female=1。"""
    X = df[FEATURES].copy()
    X["Sex"] = (X["Sex"].str.lower() == "female").astype(int)
    return X


def prep(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """特徴量に (train で決めた) 中央値補完を適用する。"""
    return features(df).fillna(medians)


def row_predict(row: np.ndarray, node: Node) -> float:
    """1 行を木に流し、葉の値 (生存確率 or u) を返す。"""
    while node.prob is None:
        node = node.left if row[node.feature] <= node.threshold else node.right
    return node.prob


def tree_predict(X: np.ndarray, node: Node) -> np.ndarray:
    return np.array([row_predict(row, node) for row in X])


def cart(X: np.ndarray, y: np.ndarray, depth: int, max_depth: int | None,
         min_samples_leaf: int, criterion: str, rng: np.random.Generator,
         max_features: int | None, cols: np.ndarray) -> Node:
    """CART を再帰構築する。criterion: 'gini' (分類) / 'mse' (回帰)。

    分割探索は累積和でベクトル化: 特徴量ごとにソートして閾値候補を一括評価する。
    """
    if (max_depth is not None and depth >= max_depth) \
            or len(y) < min_samples_leaf or len(np.unique(y)) == 1:
        return Node(prob=float(y.mean()), n=len(y))

    feats = cols if max_features is None \
        else np.sort(rng.choice(cols, size=max_features, replace=False))
    best: tuple[float, int, float] | None = None
    for j in feats:
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        ys = y[order]
        cut = np.flatnonzero(xs[1:] != xs[:-1])  # 値が変わる位置 = 閾値候補
        if len(cut) == 0:
            continue
        left = cut + 1
        right = len(ys) - left
        # sklearn と同じ min_samples_leaf: 分割後の両葉が下限を満たす候補のみ
        ok = (left >= min_samples_leaf) & (right >= min_samples_leaf)
        if not ok.any():
            continue
        left, right, cut = left[ok], right[ok], cut[ok]
        cs = np.cumsum(ys)
        lsum = cs[cut]
        rsum = cs[-1] - lsum
        if criterion == "gini":
            lp = lsum / left
            rp = rsum / right
            imp = (left * (1 - lp**2 - (1 - lp) ** 2)
                   + right * (1 - rp**2 - (1 - rp) ** 2)) / len(ys)
        else:  # mse: 左右の残差平方和
            css = np.cumsum(ys * ys)
            imp = (css[cut] - lsum**2 / left) \
                + (css[-1] - css[cut] - rsum**2 / right)
        i = int(np.argmin(imp))
        if best is None or imp[i] < best[0]:
            best = (float(imp[i]), int(j), float(xs[cut[i]]))
    if best is None:
        return Node(prob=float(y.mean()), n=len(y))
    _, j, thr = best
    mask = X[:, j] <= thr
    return Node(
        feature=j, threshold=thr,
        left=cart(X[mask], y[mask], depth + 1, max_depth, min_samples_leaf,
                  criterion, rng, max_features, cols),
        right=cart(X[~mask], y[~mask], depth + 1, max_depth, min_samples_leaf,
                   criterion, rng, max_features, cols),
        n=len(y),
    )


class RandomForest(Predictor):
    """ブートストラップ標本ごとに CART を学習し、予測は生存確率の平均。

    過学習しやすい深い木を大量に作り、その不安定性を平均で相殺する。
    """

    def __init__(self, n_estimators: int = 500, max_depth: int | None = None,
                 min_samples_leaf: int = 1, max_features: int | None = None,
                 seed: int = 0, features: list[str] | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.features = list(features) if features else list(FEATURES)
        self.trees: list[Node] = []
        self.inbag: list[np.ndarray] = []  # 木ごとの学習に使った行 (bool)
        self.medians: pd.Series = pd.Series(dtype=float)
        self.oob_scores: np.ndarray | None = None  # 学習行の OOB 生存確率
        self.oob_std: np.ndarray | None = None  # 学習行の OOB 木間ばらつき (不確実性)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """self.features の列を組み立てる (Sex は female=1)。"""
        X = df[self.features].copy()
        X["Sex"] = (X["Sex"].str.lower() == "female").astype(int)
        return X

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量に (train で決めた) 中央値補完を適用する。"""
        return self._features(df).fillna(self.medians)

    def fit(self, df: pd.DataFrame) -> "RandomForest":
        y = df["Survived"].to_numpy(dtype=float)
        feat = self._features(df)
        self.medians = feat.median(numeric_only=True)  # train で固定
        X = self._prep(df).to_numpy()
        n = len(y)
        rng = np.random.default_rng(self.seed)
        cols = np.arange(X.shape[1])

        self.trees = []
        self.inbag = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n)  # ブートストラップ (復元抽出)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            self.inbag.append(mask)
            self.trees.append(cart(X[idx], y[idx], 0, self.max_depth,
                                   self.min_samples_leaf, "gini", rng,
                                   self.max_features, cols))

        # OOB: 木ごとに「学習に使わなかった行」だけで予測し、平均とばらつきを取る
        oob_sum = np.zeros(n)
        oob_sq = np.zeros(n)
        oob_cnt = np.zeros(n, dtype=int)
        for t, tree in enumerate(self.trees):
            oob = ~self.inbag[t]
            if not oob.any():
                continue
            p = tree_predict(X, tree)[oob]
            oob_sum[oob] += p
            oob_sq[oob] += p * p
            oob_cnt[oob] += 1
        # どの木にも OOB にならなかった行 (稀) は全体アンサンブルで補完
        full = self.predict_score(df)
        oob_mean = np.where(oob_cnt > 0, oob_sum / np.maximum(oob_cnt, 1), full)
        oob_var = np.where(oob_cnt > 0,
                           oob_sq / np.maximum(oob_cnt, 1) - oob_mean**2, 0.0)
        self.oob_scores = oob_mean
        self.oob_std = np.sqrt(np.maximum(oob_var, 0.0))
        return self

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prep(df).to_numpy()
        scores = np.zeros(len(X))
        for tree in self.trees:
            scores += tree_predict(X, tree)
        return scores / len(self.trees)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series((self.predict_score(df) >= 0.5).astype(int), index=df.index)


class RegressionTree:
    """浅い回帰木 (SSE 分割)。g: x → u の領域モデリング用。"""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 10, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        rng = np.random.default_rng(self.seed)
        cols = np.arange(X.shape[1])
        self.root = cart(X, y, 0, self.max_depth, self.min_samples_leaf,
                         "mse", rng, None, cols)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return tree_predict(X, self.root)


def leaf_regions(node: Node, X: np.ndarray, feature_names: list[str]
                 ) -> list[tuple[str, np.ndarray]]:
    """葉ごとに (ルール文字列, その領域に落ちた行インデックス) を集める。"""
    out: list[tuple[str, np.ndarray]] = []

    def walk(n: Node, rows: np.ndarray, conds: list[str]) -> None:
        if n.prob is not None:
            out.append((" & ".join(conds) if conds else "all", rows))
            return
        m = X[rows][:, n.feature] <= n.threshold
        if feature_names[n.feature] == "Sex":
            walk(n.left, rows[m], conds + ["男"])
            walk(n.right, rows[~m], conds + ["女"])
        else:
            f = feature_names[n.feature]
            walk(n.left, rows[m], conds + [f"{f}<={n.threshold:.1f}"])
            walk(n.right, rows[~m], conds + [f"{f}>{n.threshold:.1f}"])

    walk(node, np.arange(len(X)), [])
    return out


def accuracy(model: Predictor, df: pd.DataFrame) -> float:
    return float((model.predict(df) == df["Survived"]).mean())


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    train, val = df.iloc[:712], df.iloc[712:]
    y_tr = train["Survived"].to_numpy()
    y_va = val["Survived"].to_numpy()
    med = features(train).median(numeric_only=True)
    X_tr = prep(train, med).to_numpy()
    X_va = prep(val, med).to_numpy()

    # --- 1. 1本の決定木 → Random Forest ---
    print("== 1. 1本の決定木 → Random Forest ==")
    single = DecisionTree(max_depth=712, min_samples_leaf=1).fit(train)
    rf_deep = RandomForest(n_estimators=500, max_depth=None,
                           min_samples_leaf=1, seed=0).fit(train)
    print(f"1本の決定木    (max_depth=None)  学習 {accuracy(single, train):.4f}"
          f" / 検証 {accuracy(single, val):.4f}")
    print(f"Random Forest  (500本, 同条件)   学習 {accuracy(rf_deep, train):.4f}"
          f" / 検証 {accuracy(rf_deep, val):.4f}")
    print("→ 1本の木は学習を丸暗記し検証で劣化する。RF は多数の不安定な木の"
          "平均で過学習を相殺する。")

    # --- 2. deep RF vs shallow RF の OOB 局所誤差 ---
    print("\n== 2. deep RF vs shallow RF の OOB 比較 ==")
    rf_shallow = RandomForest(n_estimators=500, max_depth=4,
                              min_samples_leaf=10, seed=0).fit(train)
    p_c = rf_deep.oob_scores     # M_c^OOB(x): 深い RF の未知データ扱い予測
    p_s = rf_shallow.oob_scores  # M_s^OOB(x)
    e_c = (p_c - y_tr) ** 2
    e_s = (p_s - y_tr) ** 2
    u = e_s - e_c  # u > 0 ⇔ 深さが効いた (shallow が外して deep が当てた)

    # M_c(x) - M_s(x): 細かい局所構造まで使ったことで予測がどれだけ変わったか
    d = p_c - p_s
    flip = (p_c >= .5) != (p_s >= .5)
    print(f"M_c - M_s: 平均 {d.mean():+.4f}, |差| 平均 {np.abs(d).mean():.4f},"
          f" 0.5 をまたいで予測が反転した人 {flip.sum()}人")
    print(f"deep    (max_depth=None, min_leaf=1)  OOB 正答率 {((p_c >= .5) == y_tr).mean():.4f}"
          f" / 平均二乗誤差 {e_c.mean():.4f}")
    print(f"shallow (max_depth=4,   min_leaf=10)  OOB 正答率 {((p_s >= .5) == y_tr).mean():.4f}"
          f" / 平均二乗誤差 {e_s.mean():.4f}")
    print(f"u = e_s - e_c: 平均 {u.mean():+.4f}, u>0 (deep が勝った領域)"
          f" {(u > 0).sum()}人 / u<0 {(u < 0).sum()}人")

    err = train.copy()
    err["p_deep"] = p_c.round(3)
    err["p_shallow"] = p_s.round(3)
    err["u"] = u.round(3)
    cols = ["u", "Survived", "p_deep", "p_shallow", "Sex", "Pclass", "Age", "Fare"]
    print("\nu>0 上位 (shallow が外し deep が当てた = 複雑さが効いた):")
    print(err.nlargest(8, "u")[cols].to_string(index=False))
    print("\nu<0 上位 (deep が外し shallow が当てた = 複雑さが裏目):")
    print(err.nsmallest(8, "u")[cols].to_string(index=False))

    err["delta"] = d.round(3)
    dcols = ["delta", "Survived", "p_deep", "p_shallow", "Sex", "Pclass", "Age", "Fare"]
    print("\n|M_c - M_s| が大きい乗客 (局所構造で予測が最も変わった, delta>0 = deep が生存寄り):")
    print(err.reindex(err["delta"].abs().sort_values(ascending=False).index)
          .head(8)[dcols].to_string(index=False))

    # --- 3. g(x) → u: 複雑さが効く領域のモデル化 ---
    print("\n== 3. g(x) → u (複雑さが効く領域) ==")
    g = RegressionTree(max_depth=3, min_samples_leaf=10, seed=0).fit(X_tr, u)
    print("葉ごとの領域と判定 (t = u 平均/標準誤差。|t|>=1.7 かつ正答率が向く方向を有意とみなす):")
    print(f"{'領域':<26}{'n':>4}{'u平均':>7}{'t':>6}{'deep正答':>9}{'shallow正答':>11}  判定")
    hurt: list[str] = []
    helped: list[str] = []
    region_rows: list[dict] = []
    for rule, rows in leaf_regions(g.root, X_tr, FEATURES):
        uu = u[rows]
        mu = uu.mean()
        sd = uu.std(ddof=1) if len(uu) > 1 else 0.0
        t = mu / (sd / np.sqrt(len(uu))) if sd > 0 else 0.0
        ad = ((p_c[rows] >= .5) == y_tr[rows]).mean()
        as_ = ((p_s[rows] >= .5) == y_tr[rows]).mean()
        if abs(t) >= 1.7 and t < 0 and as_ >= ad:
            verdict = "shallow有利 (deepで精度低下)"
            hurt.append(rule)
        elif abs(t) >= 1.7 and t > 0 and ad >= as_:
            verdict = "deep有利"
            helped.append(rule)
        else:
            verdict = "差なし"
        region_rows.append({"region": rule, "n": len(rows), "u_mean": mu,
                            "t": t, "deep_acc": ad, "shallow_acc": as_,
                            "verdict": verdict})
        print(f"{rule:<26}{len(rows):>4}{mu:+7.3f}{t:+6.2f}{ad:>9.3f}{as_:>11.3f}  {verdict}")
    if hurt:
        print(f"→ deep を使うと明確に予測精度が下がる領域: {'; '.join(hurt)}")
    if helped:
        print(f"→ deep が明確に勝つ領域: {'; '.join(helped)}")
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    out = outdir / "rf_regions.csv"
    pd.DataFrame(region_rows).to_csv(out, index=False)
    print(f"出力: {out}")

    # 乗客ごとの OOB 予測 + g の領域 (nb の専用セクション用)
    region_of = np.empty(len(X_tr), dtype=object)
    for rule, rows in leaf_regions(g.root, X_tr, FEATURES):
        region_of[rows] = rule
    pas = train.copy()
    pas["p_deep"] = p_c.round(3)
    pas["p_shallow"] = p_s.round(3)
    pas["u"] = u.round(3)
    pas["region"] = region_of
    pas[["region", "Survived", "p_deep", "p_shallow", "u",
         "Sex", "Pclass", "Age", "Fare"]].to_csv(outdir / "rf_passengers.csv",
                                                  index=False)
    print(f"出力: {outdir / 'rf_passengers.csv'}")
    g_tr = g.predict(X_tr)
    print(f"学習データで g と実際の u の相関: {np.corrcoef(g_tr, u)[0, 1]:+.3f}")

    # 検証: 保持データで deep/shallow の実予測差と g の予測を突き合わせる
    p_c_va = rf_deep.predict_score(val)
    p_s_va = rf_shallow.predict_score(val)
    u_va = (p_s_va - y_va) ** 2 - (p_c_va - y_va) ** 2
    g_va = g.predict(X_va)
    print("\n検証データでの確認 (g の予測領域ごとに実際の u 平均):")
    for label, m in [("g > 0 (deep が効く領域)", g_va > 0),
                     ("g <= 0 (shallow の領域)", g_va <= 0)]:
        acc_d = ((p_c_va[m] >= .5) == y_va[m]).mean()
        acc_s = ((p_s_va[m] >= .5) == y_va[m]).mean()
        print(f"  {label}: {m.sum()}人, 実測u平均 {u_va[m].mean():+.4f},"
              f" 正答率 deep {acc_d:.4f} / shallow {acc_s:.4f}")
    print(f"  全体: 正答率 deep {((p_c_va >= .5) == y_va).mean():.4f}"
          f" / shallow {((p_s_va >= .5) == y_va).mean():.4f}")


if __name__ == "__main__":
    main()
