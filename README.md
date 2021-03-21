# AI王 〜クイズAI日本一決定戦〜 解法
## 事前に用意するもの
- cuDNN: 7.6.5
- cuda: 10.1
- [apex](https://github.com/NVIDIA/apex)
- [推論時必要な学習済みパラメータ](https://drive.google.com/drive/folders/185aD55z77MP-1IwApDqt7ULfrYiRuNdA?usp=sharing)\
./paramsの配下に学習済みパラメータを配置してください

- ./dataの配下にaio_leaderboard.json


## データの前処理
学習のために必要なファイルはpreprocessから生成することができます．

preprocess内で以下のコードを実行すると訓練に必要なファイルが生成できます．
```
. preprocess.sh

```
## 学習
  上記の前処理で生成したデータがあれば学習できます

  ```
  例
  python train-mix-top1.py
  ```
## 推論
  inference.ipynbより推論できます


## 検索のみの実装

[https://github.com/9shikixp/aio-bm25](https://github.com/9shikixp/aio-bm25)

  
