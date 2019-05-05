import os
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime, timedelta


class DnnModel(nn.Module):
    def __init__(self, node_nums=[],):
        super(DnnModel, self).__init__()
        self.node_nums = node_nums
        if self.node_nums[-1] == 0:
            self.node_nums = self.node_nums[:-1]
        self.node_nums = [6] + self.node_nums
        self.layers = []
        for i_layer in range(1, len(self.node_nums)):
            self.layers.append(nn.Linear(self.node_nums[i_layer-1], self.node_nums[i_layer], bias=True))
            self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)
        self.out = nn.Linear(self.node_nums[-1], 1, bias=False)
        # self.layers.append(nn.Linear(self.node_nums[-1], 2, bias=True))

    def forward(self, x):
        out = self.layers(x)
        out = self.out(out)
        # return F.log_softmax(out, dim=1)
        return out


def get_ty_IDs(df, obs_name):
    OBS = df[df.agency == obs_name]
    ty_ID = OBS.ty_ID
    # print('len of ty_ID is ', len(ty_ID))
    ty_IDs = []
    ty_IDs.append(ty_ID.iloc[0])
    count = 0
    for j in range(1, len(ty_ID)):
        if ty_ID.iloc[j] == ty_IDs[count]:
            continue
        else:
            ty_IDs.append(ty_ID.iloc[j])
            count += 1
    # print('ty_IDs are ', ty_IDs)
    ty_IDs = np.sort(np.array(ty_IDs))
    # print(ty_IDs)
    return ty_IDs


def read_TC_Data(root):
    df = pd.read_csv(root)
    df[df == -999] = np.nan
    return df


def pack_each_forcast_record(inputs, train_set, agencies):
    _train_set = np.zeros([1, 6]) * np.nan
    _train_set = _train_set[0]
    print(_train_set)
    '''Loop each forecast of this time record'''
    for m in range(len(inputs)):
        if inputs['agency'].iloc[m] == 'BABJ':
            continue
        '''Loop each agency to put the data at the right position'''
        for n in range(len(agencies)):
            if inputs['agency'].iloc[m] == agencies[n]:
                _train_set[n] = inputs['24MaxWind'].iloc[m]
                print('Predict 24h time: ', inputs['time'].iloc[m])
                print('Predict 24h agencies: ', inputs['agency'].iloc[m])
                print('Predict 24h MaxWind: ', inputs['24MaxWind'].iloc[m])
    train_set.append(_train_set)
    return train_set, _train_set


def pack_trainset(ty_IDs, df, agencies, pre_hour=-24):
    count = 0
    train_set = []
    target = []
    '''Loop each TC case'''
    for j in range(0, len(ty_IDs)):
        each_typhoon = df[df.ty_ID == ty_IDs[j]]
        OBS = each_typhoon[each_typhoon.agency == 'BABJ']
        time = OBS.time
        # print(each_typhoon.agency, each_typhoon.time)
        # print(OBS)
        '''Loop each time record of every TC'''
        for k in range(len(time)):
            print('j, k = ', j, k)
            _time = time.iloc[k]
            _time = datetime(year=int(_time[0:4]), month=int(_time[5:7]), day=int(_time[8:10]), hour=int(_time[11:13]))
            # print(_time)
            print('OBS time: ', _time)
            print('OBS MaxWind: ', OBS['MaxWind'].iloc[k])
            target.append(OBS['MaxWind'].iloc[k])
            start_time = _time + timedelta(hours=pre_hour)
            # print(start_time)
            inputs = each_typhoon[each_typhoon.time == str(start_time)]
            train_set, _train_set = pack_each_forcast_record(inputs, train_set, agencies)
            print('train_set = ', _train_set)
            print('target', OBS['MaxWind'].iloc[k])
        break
    return [train_set, target]


def pack_data(save_data=True):
    df = read_TC_Data('/Users/ageliss/Documents/GitHub/1th_Ocean_Predict_Center/Typhoon_data2.csv')
    # case = df[df.ty_name == 'MAN-YI']
    agencies = ['KSLR', 'RJTD', 'BABJ', 'PGTW', 'VHHH', 'WRF', 'COAWST']
    ty_IDs = get_ty_IDs(df, 'BABJ')
    pre_hour = -24
    [train_set, target] = pack_trainset(ty_IDs, df, agencies, pre_hour=pre_hour)
    train_set = np.reshape(train_set, [-1, 6])
    target = np.reshape(target, -1)
    if save_data:
        np.save('packed_trainset'+str(pre_hour)+'.npy', np.array([np.transpose(train_set), target]))
        # np.save('packed_target' + str(pre_hour) + '.npy', target)
    print('END Packing data: reshape = ', train_set)
    return [train_set, target]


def main(load_data=True):
    if load_data:
        print('loading ...')
        train_sets, targets = np.load('./packed_trainset-24.npy')
        train_sets = train_sets.transpose()
        train_sets[np.isnan(train_sets)] = 0
        print('train_sets = ', train_sets)
        print('targets = ', targets)
    else:
        [train_sets, targets] = pack_data(save_data=True)
        print('Ending pack data')
    model = DnnModel(node_nums=[20, 0])
    print(model)
    x_test = np.random.rand(10, 6)
    y_test = np.random.rand(10)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.)
    train_set, target = Variable(torch.from_numpy(train_sets).float()), \
                        Variable(torch.from_numpy(targets).float())
    x_test, y_test = Variable(torch.from_numpy(x_test).float()), \
                        Variable(torch.from_numpy(y_test).float())
    for epoch in range(1000):
        model.eval()
        y_pre_test = model(x_test)
        # print(y_pre_test.reshape(-1))
        loss_test = float(F.mse_loss(y_pre_test.reshape(-1), y_test).item())

        model.train()
        optimizer.zero_grad()
        output = model(train_set)
        loss = F.mse_loss(output.reshape(-1), target)
        print(loss)
        loss.backward()
        print('below is weight0:')
        print(model.state_dict()['layers.0.weight'])
        print('below is weight2:')
        print(model.state_dict()['out.weight'])
        optimizer.step()
    print('predicted output = ', output.reshape(-1))
    print('Ground Truth = ', target)
    print('Predicted MSE = ', loss.item())
    print('Original_Mean_Value_Method MSE = ', ((np.nanmean(train_sets_withNone, axis=1) - targets) ** 2).mean(axis=0))


if __name__ == '__main__':
    main(load_data=True)
    '''Try Xgboost'''




