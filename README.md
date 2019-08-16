# DaConA
This is the implementation (PyTorch Version) of DaConA which is submitted to BigData19.

All codes are written by Python 3.6.

Requirements: numpy, pytorch(>=0.4) with CUDA

## Abstract
Given a sparse rating matrix and an auxiliary matrix of users or items, how can we effectively leverage such matrices and predict missing values in the rating matrix accurately? Predicting rating values is a crucial problem in recommendation because users want to be served items that they will give high ratings. Many previous studies proved that utilizing the additional information with rating data is helpful to improve the performance. 
However, existing methods are limited in that 1) they have restricted capability of expressing independence information of users or items, 2) they do not consider the fact that data contexts of rating auxiliary matrices are different, and 3) they assume the relation between a user and an item is linear. 

In this paper, we propose DaConA, a neural network based method for recommendation with a rating matrix and an auxiliary matrix. DaConA is designed with the following three main ideas. First, DaConA represents each entity with a latent interaction vector and latent independence vector. Unlike previous methods, both of the two vectors are not limited in size. Second, we propose a data context adaptation layer to extract appropriate features for different data contexts. Lastly, while previous matrix factorization based methods predict missing values through the inner-product of latent vectors, DaConA learns a non-linear function of them via a neural network. We show that DaConA is a generalized algorithm including the standard matrix factorization and the collective matrix factorization as special cases. Through extensive experiments on real-world datasets, we show that DaConA provides the state-of-the-art performance in real-world datasets.

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
