# Titanic: Machine Learning from Disaster

## 問題

1912 年に沈没したタイタニック号の乗客情報（train.csv, 891人）から、各乗客が生存したかどうか（Survived = 0/1）を予測する分類問題。

test.csv（418人）の生存を予測し、提出フォーマットに従って提出する。

## データ

| ファイル | 説明 |
|---|---|
| `data/train.csv` | 訓練データ。目的変数 `Survived` を含む |
| `data/test.csv` | 予測対象。`Survived` を含まない |
| `data/gender_submission.csv` | サンプル提出（女性=生存）。提出フォーマットの参考 |

### 列の意味

- `PassengerId`: 乗客ID
- `Survived`: 生存 (1) / 死亡 (0) — 目的変数（train のみ）
- `Pclass`: 客室クラス (1=上等, 2, 3=下等)
- `Name`: 氏名（敬称を含む）
- `Sex`: 性別
- `Age`: 年齢（欠損あり）
- `SibSp`: 同乗の兄弟・配偶者数
- `Parch`: 同乗の親・子供数
- `Ticket`: チケット番号
- `Fare`: 運賃（test に欠損1件）
- `Cabin`: 客室番号（欠損多数）
- `Embarked`: 乗船港 (C/Q/S)

## 評価

提出は Accuracy（正解率）で評価される。順位表の上位はおよそ 0.83 前後。

## 提出フォーマット

`PassengerId,Survived` の2列、418行。`data/gender_submission.csv` が Kaggle 公式のサンプル。

## 環境

Python 3.14 + uv 管理。必要なパッケージはインストール済み。

```bash
uv run jupyter notebook
```
