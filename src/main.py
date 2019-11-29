"""
Data Context Adaptation for Accurate Recommendation with Additional Information
"""
import random
import time

import click
import numpy as np
import torch

from model import *
from model import DaConA


def read_dataset_origin(trn_file, test_file, aux_file, type):
    """
    Read the given datasets and convert them to numpy-arrays and dictionaries
    :param trn_file: Training file
    :param test_file: Test file
    :param aux_file: Auxiliary file
    :param type: Type of the auxiliary file (user or item)
    :return: Numpy-arrays of trn, val, test, aux and dictionaries for user, item, aux
    """
    users = np.array([], dtype=np.int)
    items = np.array([], dtype=np.int)

    # read all data files
    Dtrn = np.loadtxt(trn_file, delimiter='\t').T
    Dtest = np.loadtxt(test_file, delimiter='\t').T
    Daux = np.loadtxt(aux_file, delimiter='\t').T

    # for user dictionary
    users = np.append(users, Dtrn[0].astype(int))
    users = np.append(users, Dtest[0].astype(int))

    if type == 'user':
        users = np.append(users, Daux[0].astype(int))
    users = np.unique(users)
    user_dict = make_dictionary(users)

    # for item dictionary
    items = np.append(items, Dtrn[1].astype(int))
    items = np.append(items, Dtest[1].astype(int))
    if type == 'item':
        items = np.append(items, Daux[0].astype(int))
    items = np.unique(items)
    item_dict = make_dictionary(items)

    # for aux dictionary
    aux_dict = make_dictionary(np.unique(Daux[1]))

    # Remap the ids from raw_id to dict_id
    for i in range(Dtrn.shape[1]):
        Dtrn[0][i] = user_dict[Dtrn[0][i]]
        Dtrn[1][i] = item_dict[Dtrn[1][i]]

    for i in range(Dtest.shape[1]):
        Dtest[0][i] = user_dict[Dtest[0][i]]
        Dtest[1][i] = item_dict[Dtest[1][i]]

    if type == 'user':
        for i in range(Daux.shape[1]):
            Daux[0][i] = user_dict[Daux[0][i]]
            Daux[1][i] = aux_dict[Daux[1][i]]
    elif type == 'item':
        for i in range(Daux.shape[1]):
            Daux[0][i] = item_dict[Daux[0][i]]
            Daux[1][i] = aux_dict[Daux[1][i]]

    return Dtrn, Dtest, Daux, user_dict, item_dict, aux_dict


def make_dictionary(array):
    """
    Make dictionary for the given array
    :param array: Numpy array to be changed as a dictionary
    :return: A dictionary that made from the given array
    """
    dict={}
    array = array.astype(int)
    for i in range(array.shape[0]):
        dict[array[i]] = i
    return dict


@click.command()
@click.option('--dataset', type=str, default='ml-100k')
@click.option('--datapath', type=str, default='../data/')
@click.option('--aux_type', type=click.Choice(TYPES), default='item',
              help='user/item')
@click.option('--dim_l', type=int, default=10)
@click.option('--dim_s', type=int, default=10)
@click.option('--n_layers', type=int, default=3,
              help='Number of layers (network is constructed in tower pattern)')
@click.option('--batch_size', type=int, default=1024)
@click.option('--lr', type=float, default=1e-3,
              help='Learning rate')
@click.option('--decay', type=float, default=1e-5,
              help='Regularization parameter')
@click.option('--alpha', type=float, default=0.9,
              help='information importance')
@click.option('--epoch', type=int, default=10000,
              help='Number of iteration')
@click.option('--seed', type=int, default=None)
@click.option('--early_stop', type=bool, default=False)
@click.option('--stop_iter', type=int, default=20)
def main(dataset, datapath, aux_type, dim_l, dim_s, n_layers, batch_size,
         lr, decay, alpha, epoch, seed, early_stop, stop_iter):
    """
    Process:
    1) Read a dataset
    2) Define a model with given hyper-parameters
    3) Than the model
    """
    # set random seed
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    datapath += dataset + '/'

    trn_file = datapath + 'trn80_shuffled.tsv'
    test_file = datapath + 'test80_shuffled.tsv'
    aux_file = datapath + 'aux.tsv'

    print(f'random seed:{seed}')
    print('------------------- Dataset information -------------------')
    print('Dataset: {}'.format(dataset))
    print('Type of auxiliary data: {}'.format(aux_type))
    print('Val file: {}'.format(test_file))
    print('Test file: {}'.format(test_file))

    # Read dataset and make dictionaries
    start_time = time.time()
    Dtrain, Dtest, Daux, user_dict, item_dict, aux_dict =\
        read_dataset_origin(trn_file, test_file, aux_file, aux_type)

    n_users = len(user_dict)
    n_items = len(item_dict)
    n_aux = len(aux_dict)

    print('n_users: {}'.format(n_users))
    print('n_items: {}'.format(n_items))
    print('n_aux: {}'.format(n_aux))
    print('Dtrain: {}'.format(Dtrain.shape))
    print('Dtest: {}'.format(Dtest.shape))
    print('Daux: {}'.format(Daux.shape))
    num_ratings = Dtrain.shape[1] + + Dtest.shape[1]
    print('Density of rating matrix: {0:.4f}%'
          .format(float(100 * num_ratings / (n_users * n_items))))
    num_aux = Daux.shape[1]
    total = 0
    if aux_type == 'item':
        total = n_items * n_aux
    else:
        total = n_users * n_aux
    print('Density of auxiliary matrix: {0:.4f}%'
          .format(float(100 * num_aux / total)))
    print('Data loaded: {0:.2f} sec'.format(time.time() - start_time))

    # Initialize a model
    model = DaConA.DaConA(n_users, n_items, n_aux, aux_type,
                          dim_l, dim_s, n_layers,
                          lr, decay)
    model.cuda()
    print('------------------- Model information -------------------')
    print(model)

    # Train the model
    if dataset in ['ciao_u', 'ciao_i', 'epinions', 'ml-1m', 'ml-100k']:
        min = 1.0
        max = 5.0
    elif dataset in ['filmtrust']:
        min = 0.5
        max = 4.0
    model.do_learn(Dtrain, Dtest, Daux, epoch, batch_size,
                   alpha, min, max, early_stop, stop_iter)


if __name__=='__main__':
    main()
