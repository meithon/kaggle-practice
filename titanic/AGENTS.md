# AGENTS.md

## Quarto + Plotly notebook (matrix.ipynb 等)

Quarto は notebook 出力の `application/vnd.plotly.v1+json` mime type を描画できない
(`plotly_mimetype` レンダラが保存するデータのみの JSON で、Quarto は非対応 → HTML に
「Unable to display output for mime type(s)」が出る)。

**対策 (必須):**
- plotly セルでは必ず `notebook` レンダラを使う:
  ```python
  import plotly.io as pio
  pio.renderers.default = "notebook"  # text/html を保存 (Quarto が描画できる)
  ```
- 明示指定も `fig.show(renderer="notebook")` にすること。`"plotly_mimetype"` は禁止。
- plotly セルを変更したら **nbclient 等で notebook を再実行してから** render/preview する
  (`quarto preview`/`quarto render` は保存済み出力を使う)。
- preview: `quarto preview matrix.ipynb --no-browser` (http://localhost:4321)
