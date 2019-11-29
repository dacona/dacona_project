# DaConA
This project is a PyTorch implementation of "Data Context Adaptation for Accurate Recommendation with Additional Information" (DaConA), which is submitted to Samsung Humantech 2019.
This paper proposes a novel approach for data context-aware recommendation, where additional information is given as well as ratings.

## Abstract
Given a sparse rating matrix and an auxiliary matrix of users or items, how can we accurately predict missing ratings considering different data contexts of entities? Many previous studies proved that utilizing the additional information with rating data is helpful to improve the performance. However, existing methods are limited in that 1) they ignore the fact that data contexts of rating and auxiliary matrices are different, 2) they have restricted capability of expressing independence information of users or items, and 3) they assume the relation between a user and an item is linear.

We propose DaConA, a neural network based method for recommendation with a rating matrix and an auxiliary matrix. DaConA is designed with the following three main ideas. First, we propose a data context adaptation layer to extract pertinent features for different data contexts. Second, DaConA represents each entity with latent interaction vector and latent independence vector. Unlike previous methods, both of the two vectors are not limited in size. Lastly, while previous matrix factorization based methods predict missing values through the inner-product of latent vectors, DaConA learns a non-linear function of them via a neural network. We show that DaConA is a generalized algorithm including the standard matrix factorization and the collective matrix factorization as special cases. Through comprehensive experiments on real-world datasets, we show that DaConA provides the state-of-the-art accuracy. 

## Prerequisites 
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [PyTorch(>=0.4)](https://pytorch.org)
- [Click(>=7.0)](https://click.palletsprojects.com)
- [NumPy(>=1.14)](https://numpy.org)

## How to Run
  - Run (in `./src/` directory): `python main.py  [Options]`

## Options

| Option  | Description | Choice  |
| ------  | ----------- | ------  |
|`--dataset` | dataset to train/test | `ciao_i`, `ciao_u`, `epinions`, `filmtrust`, `ml-1m`, or `ml-100k`|
|`--aux_type` | type of auxiliary information | `user` or `item`|
|`--dim_l` | dimension of the predictive factor | For the optimal value, see table 5 in the paper|
|`--dim_s` | dimension of latent independence factor | For the optimal value, see table 5 in the paper|
|`--batch_size` | size of mini-batch | `1024` (default)|
|`--lr` | learning rate | For the optimal value, see table 5 in the paper|
|`--decay` | weight decay of l2 regularizer | For the optimal value, see table 5 in the paper|
|`--alpha` | balance parameter for rating matrix | For the optimal value, see table 5 in the paper|
|`--beta` | balance parameter for auxiliary matrix | For the optimal value, see table 5 in the paper|
|`--epoch` | maximum number of epoch | `10000` (default)|
|`--seed` | random seed for numpy and pytorch | `None` (the random value is automatically assigned)|
|`--early_stop` | whether to stop early when the stop condition is met | `True`|
|`--stop_iter` | number of epochs that tolerate rising RMSE | `20`|
