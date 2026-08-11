"""k-fold CV で領域地図 (deep vs shallow) が再現するか検証する。

random_forest.py の 1 分割 (712/179, seed=0) で「成人・低運賃では shallow 有利、
子供では deep 有利」という領域判定が出た。これがデータの切り方に依存しない
本物かを、stratified k-fold CV で測る。

設計:
- 各 fold: 残り 4/5 で deep/shallow RF と g を再学習 → holdout 1/5 で
  g の各領域の実際の正答率差 (shallow - deep) を確認。
- モデル側の seed は固定し、分割 (fold 割り当て) だけを変える:
  地図が動いた原因はデータの切り方にある、と帰属できるように。
- 判定が holdout でも同じ向きなら頑健、符号が割れたらノイズ。

使い方:
    uv run python src/rf_cv.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from random_forest import (RandomForest, RegressionTree, features, prep,
                           FEATURES, leaf_regions)


def stratified_kfold(n: int, y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """クラス比率を保ったまま n 人を k グループに分ける (stratified)。"""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        for i, j in enumerate(idx):
            folds[i % k].append(int(j))
    return [np.array(sorted(f)) for f in folds]


def verdict_of(u: np.ndarray, deep_acc: float, shallow_acc: float) -> str:
    """random_forest.py と同じ判定 (|t|>=1.7 かつ正答率が向く方向)。"""
    mu = u.mean()
    sd = u.std(ddof=1) if len(u) > 1 else 0.0
    t = mu / (sd / np.sqrt(len(u))) if sd > 0 else 0.0
    if abs(t) >= 1.7 and t < 0 and shallow_acc >= deep_acc:
        return "shallow有利 (deepで精度低下)"
    if abs(t) >= 1.7 and t > 0 and deep_acc >= shallow_acc:
        return "deep有利"
    return "差なし"


def acc_delta(pa: np.ndarray, pb: np.ndarray, y: np.ndarray) -> float:
    """shallow(pb) - deep(pa) の正答率差。正 = shallow が良い。"""
    return float(((pb >= .5) == y).mean() - ((pa >= .5) == y).mean())


def main() -> None:
    df = pd.read_csv(Path("data/train.csv"))
    y = df["Survived"].to_numpy()
    med = features(df).median(numeric_only=True)
    X = prep(df, med).to_numpy()
    n = len(df)

    k, seeds = 5, [0, 1, 2]
    overall: list[float] = []       # holdout 全体の正答率差 (shallow - deep)
    sh_leaf: list[tuple[float, int]] = []  # g が shallow有利 と判定した葉の (Δ, n)
    dp_leaf: list[tuple[float, int]] = []  # g が deep有利 と判定した葉の (Δ, n)
    adult_lo: list[float] = []      # 成人低運賃 (Age>11, Fare<=13.9) holdout Δ
    child: list[float] = []         # 子供 (Age<=11) holdout Δ
    fold_rows: list[dict] = []      # CSV 用 (seed, fold, 各 Δ)

    print(f"== k-fold CV (k={k}, stratified, seeds={seeds}) ==")
    for seed in seeds:
        for fi, val_idx in enumerate(stratified_kfold(n, y, k, seed)):
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            train = df.iloc[tr_idx]
            X_tr, X_va = X[tr_idx], X[val_idx]
            y_tr, y_va = y[tr_idx], y[val_idx]
            df_va = df.iloc[val_idx]

            rf_deep = RandomForest(n_estimators=500, max_depth=None,
                                   min_samples_leaf=1, seed=0).fit(train)
            rf_shallow = RandomForest(n_estimators=500, max_depth=4,
                                      min_samples_leaf=10, seed=0).fit(train)
            p_c_va = rf_deep.predict_score(df_va)
            p_s_va = rf_shallow.predict_score(df_va)

            d_all = acc_delta(p_c_va, p_s_va, y_va)
            overall.append(d_all)

            # 固定領域 (元の g の主張) を holdout で評価
            age, fare = X_va[:, 2], X_va[:, 3]  # Age/Fare は中央値補完済み
            mask_a = (age > 11) & (fare <= 13.9)
            mask_c = age <= 11
            d_adult = acc_delta(p_c_va[mask_a], p_s_va[mask_a], y_va[mask_a])
            d_child = acc_delta(p_c_va[mask_c], p_s_va[mask_c], y_va[mask_c])
            adult_lo.append(d_adult)
            child.append(d_child)
            fold_rows.append({"seed": seed, "fold": fi,
                              "overall_pt": round(d_all * 100, 2),
                              "adult_lo_pt": round(d_adult * 100, 2),
                              "child_pt": round(d_child * 100, 2)})

            # g を再学習し、各葉の train 判定 vs holdout Δ を集計
            u_tr = (rf_shallow.oob_scores - y_tr) ** 2 \
                - (rf_deep.oob_scores - y_tr) ** 2
            g = RegressionTree(max_depth=3, min_samples_leaf=10, seed=0).fit(X_tr, u_tr)
            for (tr_rule, tr_rows), (va_rule, va_rows) in zip(
                    leaf_regions(g.root, X_tr, FEATURES),
                    leaf_regions(g.root, X_va, FEATURES)):
                if len(va_rows) == 0:
                    continue
                v = verdict_of(
                    u_tr[tr_rows],
                    ((rf_deep.oob_scores[tr_rows] >= .5) == y_tr[tr_rows]).mean(),
                    ((rf_shallow.oob_scores[tr_rows] >= .5) == y_tr[tr_rows]).mean(),
                )
                d = acc_delta(p_c_va[va_rows], p_s_va[va_rows], y_va[va_rows])
                if v == "shallow有利 (deepで精度低下)":
                    sh_leaf.append((d, len(va_rows)))
                elif v == "deep有利":
                    dp_leaf.append((d, len(va_rows)))
            print(f"seed {seed} fold {fi}: 全体 Δ {d_all*100:+.1f}pt | "
                  f"成人低運賃 Δ {d_adult*100:+.1f}pt | "
                  f"子供 Δ {d_child*100:+.1f}pt")

    # 葉ごとの再現率 (判定どおりの割合)
    def match_pct(rows: list[tuple[float, int]], expected_pos: bool) -> float:
        if not rows:
            return float("nan")
        ds = np.array([d for d, _ in rows])
        return float(((ds > 0) if expected_pos else (ds < 0)).mean() * 100)

    a = np.array(adult_lo)
    c = np.array(child)
    summary = pd.DataFrame({
        "metric": ["overall_mean_pt", "overall_shallow_wins",
                   "adult_lo_mean_pt", "adult_lo_shallow_wins",
                   "child_mean_pt", "child_deep_wins",
                   "shallow_leaf_repro_pct", "deep_leaf_repro_pct"],
        "value": [round(np.mean(overall) * 100, 2), sum(d > 0 for d in overall),
                  round(a.mean() * 100, 2), sum(d > 0 for d in a),
                  round(c.mean() * 100, 2), sum(d < 0 for d in c),
                  round(match_pct(sh_leaf, True), 1),
                  round(match_pct(dp_leaf, False), 1)],
    })

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(outdir / "rf_cv.csv", index=False)
    summary.to_csv(outdir / "rf_cv_summary.csv", index=False)
    print(f"出力: {outdir / 'rf_cv.csv'}, {outdir / 'rf_cv_summary.csv'}")

    print("\n== 集計 ==")
    print(f"全体 holdout 正答率差 (shallow-deep): {np.mean(overall)*100:+.1f}pt, "
          f"shallow有利 の分割 {sum(d > 0 for d in overall)}/{len(overall)}")

    def report(name: str, rows: list[tuple[float, int]], expected_pos: bool) -> None:
        if not rows:
            print(f"  g が{name}と判定した葉: 該当なし")
            return
        ds = np.array([d for d, _ in rows])
        ns = np.array([nd for _, nd in rows])
        match = (ds > 0) if expected_pos else (ds < 0)
        print(f"  g が{name}と判定した葉: holdout 平均 Δ "
              f"{np.average(ds, weights=ns)*100:+.1f}pt, 判定通り {match.mean()*100:.0f}% "
              f"({int(ns.sum())} 人検証)")

    report("shallow有利", sh_leaf, True)
    report("deep有利", dp_leaf, False)
    a = np.array(adult_lo)
    c = np.array(child)
    print(f"成人低運賃 (Age>11, Fare<=13.9) holdout Δ: 平均 {a.mean()*100:+.1f}pt, "
          f"shallow有利 {sum(d > 0 for d in a)}/{len(a)} 分割")
    print(f"子供 (Age<=11) holdout Δ: 平均 {c.mean()*100:+.1f}pt, "
          f"deep有利 {sum(d < 0 for d in c)}/{len(c)} 分割")


if __name__ == "__main__":
    main()
