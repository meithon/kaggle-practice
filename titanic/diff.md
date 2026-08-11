**Random Forestは君の発想の一部をすでにやってるけど、同じものではない。**

まずRandom Forestを一言でいうと、

> **過学習しやすい決定木を大量に作って、平均することで過学習を減らすモデル**

だと思えばいい。

1本の深い決定木 (T_1) はかなり過学習する。

Random Forestではデータを少しずつ変えて、

[
T_1(x),T_2(x),\dots,T_{500}(x)
]

みたいに大量の木を作って、

[
f_{\mathrm{RF}}(x)
==================

\frac1{500}\sum_j T_j(x)
]

とする。

各木が拾った偶然のノイズはバラバラなので、平均すると打ち消される。一方で本当に存在するパターンは多くの木が共通して拾うので残る。

つまり、

[
\boxed{\text{Random Forest}=\text{過学習を平均で消す}}
]

という発想。

---

君が考えてるのは違って、

[
\boxed{\text{過学習していない部分だけ選んで残す}}
]

に近い。

例えば、

```text
                  領域A       領域B
深いモデル          強い         弱い
浅いモデル          普通         強い
```

だったとする。

Random Forestは基本的に、

> 全領域で大量の木を平均する

だけ。

君の方法なら、

```text
領域A → 深いモデル
領域B → 浅いモデル
```

と変える。

だから**Random Forestを使えば不要、ではない。**

むしろ組み合わせられる。

### Titanicならかなり面白い構成ができる

例えば、

[
M_c=\text{深いRandom Forest}
]

[
M_s=\text{浅いRandom Forest}
]

とする。

深い方：

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
)
```

浅い方：

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=10,
)
```

そして、

[
M_c(x)-M_s(x)
]

を見る。

これなら、

> **細かい局所構造まで使ったことで予測がどれだけ変わったか**

が取れる。

さらにRandom Forestには君のアイデアと相性がいい仕組みがある。

**OOB prediction（Out-of-Bag prediction）**。

各木は全訓練データを使わないので、ある乗客 (x_i) を学習に使っていない木だけで

[
\hat y_i^{OOB}
]

を計算できる。

つまりCVを毎回回さなくても、ある程度

> 「この乗客を未知データとして扱ったとき、このモデルはどう予測するか」

が得られる。

だから、

[
e_c(x_i)
========

L(y_i,M_c^{OOB}(x_i))
]

[
e_s(x_i)
========

L(y_i,M_s^{OOB}(x_i))
]

から

[
u_i=e_s(x_i)-e_c(x_i)
]

を作れる。

これを教師にして、

[
g(x)\rightarrow u
]

を学習すれば、まさに

> **Random Forestの複雑さが有効になる領域**

をモデル化できる。

---

そしてRandom Forestを理解する上で一番重要なのはここ。

**Random Forestは「過学習しない決定木」ではない。**

むしろ、

[
\boxed{
\text{過学習する木を大量に作って、その不安定性を平均で相殺する}
}
]

というかなり面白い設計。

だから今考えている「わざと複雑モデルを作る」という発想とはむしろ親戚。

Random Forest：

[
\text{複雑モデル群}
\rightarrow
\text{平均して安定化}
]

君の案：

[
\text{複雑モデル}
\rightarrow
\text{どこなら信用できるか学習}
\rightarrow
\text{局所的に利用}
]

この違い。

**提案としては、Titanicでまず「1本の決定木 → Random Forest」が何を改善しているのか理解してから、deep RF vs shallow RFの局所誤差を見ると、今のアイデアがかなり立体的に見える。確度: 98%**
