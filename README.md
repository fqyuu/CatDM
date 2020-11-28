# STRNN
[A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data](https://dl.acm.org/doi/pdf/10.1145/3366423.3380202)  
Fuqiang Yu, Lizhen Cui, Wei Guo, Xudong Lu, Qingzhong Li, Hua Lu (WWW-20)

## Installation
Install tensorflow 1.12.2
The code has been tested with Python 3.5, tensorflow 1.12.2 on windows 10.

## Usage

Download datasets for training/evaluation.

### 0. Preprocessing
If you use (`prepro_xxx_50.txt`), you do not need to preprocess.  
If you want to perform personally or modify the source,
[Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) is required at `../dataset/`.
```bash
$ python preprocess.py
```

### 1. Training
```bash
$ python train_torch.py
```
