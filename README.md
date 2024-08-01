
# Dynamic Importance Weighting (DIW)

[NeurIPS'20 paper: Rethinking Importance Weighting for Deep Learning under Distribution Shift](https://proceedings.neurips.cc//paper/2020/file/8b9e7ab295e87570551db122a04c6f7c-Paper.pdf).

日本語による論文読解ノートは[こちら](https://astro-notion-blog-454.pages.dev/posts/2020-NIPS-Rethinking%20Importance%20Weighting%20for%20Deep%20Learning%20under%20Distribution%20shift/)。

# Requirements

この部分はDIWの公式のレポジトリからそのまま持ってきたものである。

The code was developed and tested based on the following environment.

- python 3.8
- pytorch 1.6.0
- torchvision 0.7.0
- cudatoolkit 10.2
- cvxopt 1.2.0
- matplotlib 
- sklearn
- tqdm

# Quick start

Fashion-MNISTに、0.4の確率で誤ったラベルを付与するというLabel Noiseを考える。これを訓練とは異なるDomainのデータであるとして、これらのデータのDomain Adaptationを行う。

学習を進めると、Train DomainへのAccuracyは低いままであるが、Test DomainでのAccuracyが上昇していく様子が見て取れる。

```
python diw.py
```

# 参考

元のGitHubレポジトリは[こちら](https://github.com/TongtongFANG/DIW)。
