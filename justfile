set shell := ["bash", "-cu"]

run-model:
    uv run python ./src/main.py

# Rill ダッシュボードを起動 (http://localhost:9009)
rill:
    rill start rill --no-open --pull-env=false

preview:
    source .venv/bin/activate && quarto preview nb/matrix.ipynb --execute --port 8888
