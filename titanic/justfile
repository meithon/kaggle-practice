set shell := ["bash", "-cu"]

run-model:
    uv run python ./src/main.py

# Kaggle に提出してスコアを確認: just submit [model] [message]
#   model:   tree | female (デフォルト tree, 出力は submissions/submission.csv 固定)
#   message: 提出メッセージ (デフォルト submit)
submit model="tree" message="submit":
    uv run python src/make_submission.py --model {{model}}
    uv run python -c "import pandas as pd; s = pd.read_csv('submissions/submission.csv'); assert len(s) == 418 and set(s['Survived'].unique()) <= {0, 1}, '形式不正'; print('形式OK:', len(s), '行')"
    kaggle competitions submit -c titanic -f submissions/submission.csv -m "{{message}}"
    kaggle competitions submissions -c titanic

# Rill ダッシュボードを起動 (http://localhost:9009)
rill:
    rill start rill --no-open --pull-env=false

preview file="nb/matrix.ipynb":
    source .venv/bin/activate && quarto preview {{file}} --execute --port 8888
