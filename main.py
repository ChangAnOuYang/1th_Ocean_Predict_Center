import os
import sys
import pandas as pd
import numpy as np
# import torch.nn as nn
# import torch
# from torch.nn.parameter import Parameter
# import torch.optim as optim
# from torch.autograd import Variable
# import torch.nn.functional as F
from datetime import datetime, timedelta


# class DnnModel(nn.Module):
#     def __init__(self, node_nums=[],):
#         super(DnnModel, self).__init__()
#         self.node_nums = node_nums
#         if self.node_nums[-1] == 0:
#             self.node_nums = self.node_nums[:-1]
#         self.node_nums = [2] + self.node_nums
#         self.layers = []
#         for i_layer in range(1, len(self.node_nums)):
#             self.layers.append(nn.Linear(self.node_nums[i_layer-1], self.node_nums[i_layer], bias=True))
#             self.layers.append(nn.ReLU())
#         self.layers = nn.Sequential(*self.layers)
#         self.out = nn.Linear(self.node_nums[-1], 1, bias=False)
#         # self.layers.append(nn.Linear(self.node_nums[-1], 2, bias=True))
#
#     def forward(self, x):
#         out = self.layers(x)
#         out = self.out(out)
#         # return F.log_softmax(out, dim=1)
#         return out


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


def main():
    df = read_TC_Data('/Users/ageliss/Documents/GitHub/1th_Ocean_Predict_Center/Typhoon_data2.csv')
    # case = df[df.ty_name == 'MAN-YI']
    agencies = ['KSLR', 'RJTD', 'BABJ', 'PGTW', 'VHHH', 'WRF', 'COAWST']
    ty_IDs = get_ty_IDs(df, 'BABJ')
    count = 0
    train_set = []
    target = []
    # for j in range(len(ty_IDs)):
    '''todo list: 用define模块化'''
    for j in range(0, len(ty_IDs)):
        each_typhoon = df[df.ty_ID == ty_IDs[j]]
        OBS = each_typhoon[each_typhoon.agency == 'BABJ']
        time = OBS.time
        # print(each_typhoon.agency, each_typhoon.time)
        # print(OBS)
        for k in range(len(time)):
            print('j, k = ', j, k)
            _time = time.iloc[k]
            _time = datetime(year=int(_time[0:4]), month=int(_time[5:7]), day=int(_time[8:10]), hour=int(_time[11:13]))
            # print(_time)
            print('OBS time: ', _time)
            print('OBS MaxWind: ', OBS['MaxWind'].iloc[k])
            target.append(OBS['MaxWind'].iloc[k])
            start_time = _time + timedelta(hours=-24)
            # print(start_time)
            inputs = each_typhoon[each_typhoon.time == str(start_time)]
            _train_set = np.zeros([1, 6]) * np.nan
            _train_set = _train_set[0]
            print(_train_set)
            for m in range(len(inputs)):
                if inputs['agency'].iloc[m] == 'BABJ':
                    continue
                for n in range(len(agencies)):
                    if inputs['agency'].iloc[m] == agencies[n]:
                        _train_set[n] = np.array(inputs['24MaxWind'].iloc[m])
                        print('Predict 24h time: ', inputs['time'].iloc[m])
                        print('Predict 24h agencies: ', inputs['agency'].iloc[m])
                        print('Predict 24h MaxWind: ', inputs['24MaxWind'].iloc[m])
            train_set.append(_train_set)
            print('train_set = ', train_set)
            print('target', target)
            if k == 2:
                print('reshape = ', np.reshape(train_set, [-1, 6]))
                # sys.exit()
    np.save('packed_trainset.npy', [train_set, target])
            # sys.exit()

    # agency_old = agencies[0]
    # print(agency_old)
    # index = agencies.index
    # for j in range(len(index)):
    #     if agencies[index[j]] == agency_old:
    #         continue
    #     else:
    #         agency_old = agencies[index[j]]
    #         print(agency_old)
    # for j in range(len(agencies)):
    #     a = df[df.ty_name == 'MAN-YI']
    #     try:
    #         b = a[a.agency == agencies[j]]
    #         print(b)
    #     except:
    #         pass



if __name__ == '__main__':
    main()




