# TNTUnet

TNTUnet is a semantic segmentation model adopting TNT module and referencing TranUnet framework.

references github repos:
1. TransUnet : https://github.com/Beckschen/TransUNet.git
2. transformer-in-transformer (TNT) : https://github.com/lucidrains/transformer-in-transformer.git

TNTUnet framework

## Result

<img src="./img/real.png" width="200px"></img>
<img src="./img/depth.png" width="200px"></img>
<img src="./img/mask.png" width="200px"></img>
<img src="./img/pred.png" width="200px"></img>

## Transformer in Transformer

<img src="./img/tnt.png" width="600px"></img>

Implementation of <a href="https://arxiv.org/abs/2103.00112">Transformer in Transformer</a>, pixel level attention paired with patch level attention for image classification, in Pytorch.

## Install

```bash
$ pip install -r requirements.txt
```

<!-- ## Citations

```bibtex
@misc{han2021transformer,
    title   = {Transformer in Transformer}, 
    author  = {Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
    year    = {2021},
    eprint  = {2103.00112},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
``` -->
