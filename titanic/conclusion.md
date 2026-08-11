# Random Forest 領域選択 — 結論まとめ

`diff.md` の提案 (deep と shallow の RF を比べて「複雑さが効く領域」を学習し、領域ごとに使い分ける) を実装し、k-fold CV で再現性を検証した結果のまとめ。

## やったこと

1. **スクラッチ RF 実装** — `src/random_forest.py`。ブートストラップ標本ごとに CART を学習し平均 (gini, 累積和でベクトル化)。`tree.py` の `DecisionTree` と `min_samples_leaf=1` で完全一致することを確認。
2. **deep vs shallow 比較** — deep = `RandomForest(500, max_depth=None, min_leaf=1)`、shallow = `RandomForest(500, max_depth=4, min_leaf=10)`。
3. **OOB 局所誤差** — `u(x) = e_s(x) − e_c(x)` (e = OOB 二乗誤差)。u > 0 ⇔ deep 有利。
4. **領域地図 g(x) → u** — 浅い回帰木で「複雑さが効く領域」を分割。
5. **k-fold CV で再現性検証** — `src/rf_cv.py`。stratified k=5 × 3 seed (計 15 分割)。

## 結果

### 1. 1本の決定木 → Random Forest (改善する)
| モデル | 学習 | 検証 |
|---|---|---|
| 1本の決定木 (max_depth=None) | 0.9775 | 0.7933 |
| Random Forest (500本) | 0.9775 | 0.8156 |

1 本の木は学習を丸暗記、RF は平均で過学習を相殺。

### 2. deep vs shallow (OOB, 1 分割)
- OOB 正答率: deep 0.8006 / shallow 0.7992 (ほぼ互角)
- OOB 平均二乗誤差: deep 0.1548 / shallow 0.1422 (shallow が上)
- `M_c − M_s`: 平均 +0.003, |差| 平均 0.145, 0.5 をまたいで予測反転 83 人 (11.7%)

### 3. 領域地図 (1 分割, 見かけ)
- **成人・低運賃 (Age>11, Fare≤13.9)**: 334 人 = 成人低運賃では deep で明確に精度低下 (deep 0.805→shallow 0.859 など, t<−2)
- **子供 (Age≤11)**: deep 有利 (n は 12〜17 と小さい)

この時点では「deep は子供で効き、成人低運賃では精度を下げる」ときれいに見えた。

### 4. k-fold CV で見たら (本物)
| 対象 | holdout Δ (shallow−deep) | 分割での一貫性 |
|---|---|---|
| **全体** | **−0.2pt** | shallow 有利 8/15 → **互角** |
| g の「shallow有利」葉の再現 | +0.6pt | 判定通り **36%** |
| g の「deep有利」葉の再現 | −2.7pt | 判定通り **31%** |
| **成人低運賃** (Age>11, Fare≤13.9) | **+2.8pt** | shallow 有利 **12/15** |
| **子供** (Age≤11) | −7.0pt | deep 有利 9/15 (大幅にぶれる) |

## 結論

- **領域地図 (g) の大半は過学習**。1 分割で「きれい」に見えた判定は、holdout ではコイン投げ以下 (36%/31%)。データの切り方次第で符号が逆転する。
- **deep と shallow は全体で互角**。「shallow が勝つ」も最初の分割の偶然だった。
- **唯一 CV を生き残った本物: 「成人・低運賃では浅いモデルを使う」**(12/15 分割, +2.8pt)。子供の「deep 有利」は向きは合うが n が小さく断定不能。
- **理由 (確実なのはここまで)**: この領域で deep が一貫して shallow に負ける (CV で再現)。deep はここで予測を 0/1 に尖らせ過信する (48% vs 0.3%)。**ただし「この領域が特別ノイズが多い」とは断言できない** — deep の誤答率 19% はもう一つの大領域 (Fare>14.5) と同程度で、ベース率はそちらの方がコイン投げに近い。「ノイズが多い」はメカニズムの仮説に過ぎない。

## 教訓

- 「1 回の分割で見えたパターン」は信用できない。**k-fold CV で切り方を変えて残るものだけが本物**。
- 領域選択の考え方 (複雑さが効かない場所で使わない) は原理的に成立しうるが、**Titanic では掘り出して使う価値のある領域は 1 つだけで、効果も 2〜3pt 程度**。

## 再現手順

```
uv run python src/random_forest.py   # 領域地図 + results/rf_regions.csv (約 15 秒)
uv run python src/rf_cv.py           # k-fold CV + results/rf_cv*.csv (約 3 分)
uv run jupyter nbconvert --to notebook --execute --inplace nb/matrix.ipynb
quarto render nb/matrix.ipynb        # 可視化
```
