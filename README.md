# CatDM

[A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data](https://dl.acm.org/doi/pdf/10.1145/3366423.3380202)  
Fuqiang Yu, Lizhen Cui, Wei Guo, Xudong Lu, Qingzhong Li, Hua Lu (WWW-20)

## Installation

Install tensorflow 1.12.2.

The code has been tested with Python 3.5, tensorflow 1.12.2 on windows 10.

## Data

Datasets for training/evaluation.

- [NYC](https://www.kaggle.com/chetanism/foursquare-nyc-and-tokyo-checkin-dataset/version/2#): the Foursquare check-ins in New York

- [TKY](https://www.kaggle.com/chetanism/foursquare-nyc-and-tokyo-checkin-dataset/version/2#): the Foursquare check-ins in Tokyo

## Usage

### 0. Generate Candidates

To filter POIs and reduce the search space.
```bash
$ python train.py
```
To train and evaluate Encoder 1 and Filter, we split each dataset into a training set, a validation set and a test set, here. Encoder 1 and filtering layers form a reasonable filter capable of reducing search space, i.e., reducing the number of candidates from which recommended POIs are selected finally.

Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.

### 1. Rank POI

To sort the POIs in the candidate set.

```bash
$ python train_rankpoi.py
```
Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.
