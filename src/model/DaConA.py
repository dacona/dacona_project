import itertools
import math
import random

import numpy as np
import torch
from torch.autograd import Variable

from model import *


def circle_batch(data, batch_size):
    it = itertools.cycle(data)
    while True:
        yield list(itertools.islice(it, batch_size))

class DaConA(torch.nn.Module):
    def __init__(self,
                 n_users, n_items, n_aux, aux_type,
                 dim_l, dim_s, n_layers, lr, decay):
        super(DaConA, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_aux = n_aux
        self.dim_l = dim_l
        self.dim_c = (2 ** (n_layers - 1)) * dim_l - 2 * dim_s
        self.dim_s = dim_s
        self.user_inter = torch.nn.Embedding(n_users, self.dim_c)\
            if self.dim_c > 0 else None
        self.item_inter = torch.nn.Embedding(n_items, self.dim_c)\
            if self.dim_c > 0 else None
        self.aux_inter = torch.nn.Embedding(n_aux, self.dim_c)\
            if self.dim_c > 0 else None
        self.user_indep_x = torch.nn.Embedding(n_users, self.dim_s)\
            if dim_s > 0 else None
        self.user_indep_z = torch.nn.Embedding(n_users, self.dim_s)\
            if dim_s > 0 else None
        self.item_indep_x = torch.nn.Embedding(n_items, self.dim_s)\
            if dim_s > 0 else None
        self.item_indep_y = torch.nn.Embedding(n_items, self.dim_s)\
            if dim_s > 0 else None
        self.aux_indep_yorz = torch.nn.Embedding(n_aux, self.dim_s)\
            if dim_s > 0 else None
        self.aux_type = aux_type
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr,
                                          weight_decay=decay)
        self.global_avg_main = 0.
        self.global_avg_aux = 0.

        # Define networks
        bias = True
        self.transfer_x = torch.nn.Linear(self.dim_c, self.dim_c, bias=bias)\
            if self.dim_c > 0 else None
        self.f_x = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Linear(
                    (2**step)*dim_l, (2**(step-1))*dim_l, bias=bias),
                torch.nn.Tanh()
            )for step in range(n_layers - 1, 0, -1)]
        )
        self.f_x_reg = torch.nn.Linear(dim_l, 1, bias=bias)
        self.transfer_yorz = torch.nn.Linear(self.dim_c, self.dim_c, bias=bias)\
            if self.dim_c > 0 else None
        self.f_yorz = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Linear(
                    (2**step)*dim_l, (2**(step-1))*dim_l, bias=bias),
                torch.nn.Tanh()
            ) for step in range(n_layers - 1, 0, -1)]
        )
        self.f_yorz_reg = torch.nn.Linear(dim_l, 1, bias=bias)

        # Initialize variables
        if self.dim_c > 0:
            torch.nn.init.xavier_normal_(self.user_inter.weight)
            torch.nn.init.xavier_normal_(self.item_inter.weight)
            torch.nn.init.xavier_normal_(self.aux_inter.weight)
            torch.nn.init.xavier_normal_(self.transfer_x.weight)
            torch.nn.init.xavier_normal_(self.transfer_yorz.weight)
        if self.dim_s > 0:
            torch.nn.init.xavier_normal_(self.user_indep_x.weight)
            torch.nn.init.xavier_normal_(self.user_indep_z.weight)
            torch.nn.init.xavier_normal_(self.item_indep_x.weight)
            torch.nn.init.xavier_normal_(self.item_indep_y.weight)
            torch.nn.init.xavier_normal_(self.aux_indep_yorz.weight)
        for _, l in enumerate(self.f_x):
            torch.nn.init.xavier_normal_(l[0].weight)
        for _, l in enumerate(self.f_yorz):
            torch.nn.init.xavier_normal_(l[0].weight)
        torch.nn.init.xavier_normal_(self.f_x_reg.weight)
        torch.nn.init.xavier_normal_(self.f_yorz_reg.weight)

    def forward(self, matrix_type, rows, cols):
        if matrix_type == MAIN_MATRIX:
            avg = Variable(torch.cuda.FloatTensor(
                    rows.size()[0]).fill_(self.global_avg_main),
                           requires_grad=False)
        elif matrix_type == AUX_MATRIX:
            avg = Variable(torch.cuda.FloatTensor(
                rows.size()[0]).fill_(self.global_avg_aux),
                           requires_grad=False)
        avg = avg.unsqueeze(1)

        if self.aux_type == TYPE_USER:
            if matrix_type == MAIN_MATRIX:
                factor_inter = self.transfer_x(self.user_inter(rows)) * self.transfer_x(self.item_inter(cols)) \
                    if self.dim_c > 0 else None
                if self.dim_s == 0:
                    factor = factor_inter
                else:
                    factor = torch.cat((self.user_indep_x(rows), self.item_indep_x(cols), factor_inter), 1) \
                        if self.dim_c > 0 else torch.cat((self.user_indep_x(rows), self.item_indep_x(cols)), 1)
                for _, l in enumerate(self.f_x):
                    factor = l(factor)
                pred = self.f_x_reg(factor)
                return pred + avg

            elif matrix_type == AUX_MATRIX:
                factor_inter = self.transfer_yorz(self.user_inter(rows)) * self.transfer_yorz(self.aux_inter(cols)) \
                    if self.dim_c > 0 else None
                if self.dim_s == 0:
                    factor = factor_inter
                else:
                    factor = torch.cat((self.user_indep_z(rows), self.aux_indep_yorz(cols), factor_inter), 1) \
                        if self.dim_c > 0 else torch.cat((self.user_indep_z(rows), self.aux_indep_yorz(cols)), 1)
                for _, l in enumerate(self.f_yorz):
                    factor = l(factor)
                pred = self.f_yorz_reg(factor)
                return pred + avg

        elif self.aux_type == TYPE_ITEM:
            if matrix_type == MAIN_MATRIX:
                factor_inter = self.transfer_x(self.item_inter(cols)) * self.transfer_x(self.user_inter(rows)) \
                    if self.dim_c > 0 else None
                if self.dim_s == 0:
                    factor = factor_inter
                else:
                    factor = torch.cat((self.item_indep_x(cols), self.user_indep_x(rows), factor_inter), 1) \
                        if self.dim_c > 0 else torch.cat((self.item_indep_x(cols), self.user_indep_x(rows)), 1)
                for _, l in enumerate(self.f_x):
                    factor = l(factor)
                pred = self.f_x_reg(factor)
                return pred + avg

            elif matrix_type == AUX_MATRIX:
                factor_inter = self.transfer_yorz(self.item_inter(rows)) * self.transfer_yorz(self.aux_inter(cols)) \
                    if self.dim_c > 0 else None
                if self.dim_s == 0:
                    factor = factor_inter
                else:
                    factor = torch.cat((self.item_indep_y(rows), self.aux_indep_yorz(cols), factor_inter), 1) \
                        if self.dim_c > 0 else torch.cat((self.item_indep_y(rows), self.aux_indep_yorz(cols)), 1)
                for _, l in enumerate(self.f_yorz):
                    factor = l(factor)
                pred = self.f_yorz_reg(factor)
                return pred + avg

    def do_learn(self,
                 Dtrain, Dtest, Daux, n_epochs, batch_size,
                 alpha, min, max, early_stop, stop_iter):
        min_test_rmse = math.inf
        min_test_print = ''
        iter = 0
        cos_print = ''

        self.global_avg_main = np.mean(Dtrain[2])
        self.global_avg_aux = np.mean(Daux[2])
        print("Global Avg for Main: {:.2f}".format(self.global_avg_main))
        print("Global Avg for Aux: {:.2f}".format(self.global_avg_aux))

        for epoch in range(n_epochs):
            """
            Train under main matrix
            """
            epoch_rmse = 0.
            epoch_mae = 0.

            # Make iterable index list
            idx = [i for i in range(Dtrain.shape[1])]
            random.shuffle(idx)
            idx_iter = circle_batch(idx, batch_size)
            n_batchs = int(Dtrain.shape[1] / batch_size)

            for batch in range(n_batchs):
                idx_batch = next(idx_iter)
                rows = Dtrain[0][idx_batch]
                rows = Variable(torch.cuda.LongTensor(rows))
                cols = Dtrain[1][idx_batch]
                cols = Variable(torch.cuda.LongTensor(cols))
                values = Dtrain[2][idx_batch]
                values = Variable(torch.cuda.FloatTensor(values)).unsqueeze(1)
                prediction = self.forward(MAIN_MATRIX, rows, cols)
                self.zero_grad()
                mse = self.mse(prediction, values)
                mae = self.mae(prediction, values)
                loss = mse * (1-alpha)
                loss.backward()
                self.optimizer.step()
                epoch_rmse += mse.item()
                epoch_mae += mae.item()

            print_str = 'Epoch:{}, TrnRMSE(main): {:.4f}, TrnMAE(main): {:.4f}, '.format(epoch+1, (epoch_rmse/n_batchs)**(.5), epoch_mae/n_batchs)

            """
            Train under aux matrix
            """
            epoch_rmse = 0.
            epoch_mae = 0.
            idx = [i for i in range(Daux.shape[1])]
            random.shuffle(idx)
            idx_iter = circle_batch(idx, batch_size)
            n_batchs = int(Daux.shape[1] / batch_size)

            for batch in range(n_batchs):
                idx_batch = next(idx_iter)
                rows = Daux[0][idx_batch]
                rows = Variable(torch.cuda.LongTensor(rows))
                cols = Daux[1][idx_batch]
                cols = Variable(torch.cuda.LongTensor(cols))
                values = Daux[2][idx_batch]
                values = Variable(torch.cuda.FloatTensor(values)).unsqueeze(1)
                prediction = self.forward(AUX_MATRIX, rows, cols)
                self.zero_grad()
                mse = self.mse(prediction, values)
                mae = self.mae(prediction, values)
                loss = mse * alpha
                loss.backward()
                self.optimizer.step()
                epoch_rmse += mse.item()
                epoch_mae += mae.item()

            print_str += 'TrnRMSE(aux): {:.4f}, TrnMAE(aux): {:.4f}, '.format((epoch_rmse/n_batchs)**(.5), epoch_mae/n_batchs)

            # Test error
            rows = Variable(torch.cuda.LongTensor(Dtest[0]))
            cols = Variable(torch.cuda.LongTensor(Dtest[1]))
            values = Variable(torch.cuda.FloatTensor(Dtest[2])).unsqueeze(1)
            prediction = self.forward(MAIN_MATRIX, rows, cols)
            prediction = torch.clamp(prediction, min=min, max=max)
            mse = self.mse(prediction, values)
            mae = self.mae(prediction, values)
            print_str += 'TestRMSE: {:.4f}, TestMAE: {:.4f}'.format(mse.item()**.5, mae.item())
            print(print_str, end=' / ')

            iter += 1
            if mse.item() < min_test_rmse:
                min_test_rmse = mse.item()
                min_test_print = print_str
                iter = 0
            print(min_test_print)

            if iter > stop_iter and early_stop:
                break
