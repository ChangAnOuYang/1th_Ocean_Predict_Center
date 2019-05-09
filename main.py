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
import matplotlib.pyplot as plt
import xgboost as xgb


class DnnModel(nn.Module):
    def __init__(self, node_nums=[],):
        super(DnnModel, self).__init__()
        self.node_nums = node_nums
        if self.node_nums[-1] == 0:
            self.node_nums = self.node_nums[:-1]
        self.node_nums = [4] + self.node_nums
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


def XGBoost_model(X_train, y_train):
    # dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
    # dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
    # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # num_round = 2
    # bst = xgb.train(param, dtrain, num_round)
    # make prediction
    # preds = bst.predict(dtest)
    # sys.exit()
    # params = {
    #     'booster': 'gbtree',
    #     'objective': 'reg:linear',
    #     'num_class': 3,
    #     'gamma': 0.1,
    #     'max_depth': 6,
    #     'lambda': 2,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.3,
    #     'min_child_weight': 3,
    #     'silent': 1,
    #     'eta': 0.1,
    #     'seed': 1000,
    #     'nthread': 4,
    # }
    # plst = params.items()
    dtrain = xgb.DMatrix(data=X_train, label=y_train, missing=-999)
    print(X_train)
    print(y_train)
    num_rounds = 500
    xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=1.0, learning_rate=0.01,
                              max_depth=50, alpha=0.1, n_estimators=1000)
    xg_reg.fit(X_train, y_train)
    # xgb.plot_importance(xg_reg)
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.show()
    return xg_reg


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
    _train_set = np.zeros([1, 7]) * np.nan
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
        # break
    return [train_set, target]


def pack_data(save_data=True):
    df = read_TC_Data('/Users/ageliss/Documents/GitHub/1th_Ocean_Predict_Center/Typhoon_data2.csv')
    # case = df[df.ty_name == 'MAN-YI']
    agencies = ['KSLR', 'RJTD', 'BABJ', 'PGTW', 'VHHH', 'WRF', 'COAWST']
    ty_IDs = get_ty_IDs(df, 'BABJ')
    pre_hour = -24
    [train_set, target] = pack_trainset(ty_IDs, df, agencies, pre_hour=pre_hour)
    train_set = np.reshape(train_set, [-1, 7])
    target = np.reshape(target, -1)
    if save_data:
        np.save('packed_trainset'+str(pre_hour)+'.npy', np.array([np.transpose(train_set), target]))
        # np.save('packed_target' + str(pre_hour) + '.npy', target)
    print('END Packing data: reshape = ', train_set)
    return [train_set, target]


def remove_allnan_produce_testset(train_sets, targets, model_type):
    '''Remove those trainsets without useful information'''
    standard = np.random.rand(len(targets))
    b_good = [0, 1, 3, 4]  # only these columns contain valuable information
    if model_type == 'bp':
        fill_mean = np.nanmean(train_sets, axis=0)
        print('fill_nanmean = ', np.nanmean(train_sets, axis=0))
        print('fill_mean_target = ', np.nanmean(targets, axis=0))
    elif model_type == 'xgboost':
        fill_mean = np.ones(6) * -999
        print('fill_nanmean = ', fill_mean)
        print('fill_mean_target = ', -999)

    for j in range(len(targets)):
        mean_value = np.nanmean(train_sets[j]) + targets[j]
        if ~np.isnan(mean_value):
            ori = train_sets[j].copy()
            for k in range(len(train_sets[j])):
                if np.isnan(train_sets[j][k]):
                    train_sets[j][k] = fill_mean[k]
            if j == 0:
                _train_sets = train_sets[j][b_good]
                _targets = targets[j]
                _test_sets = train_sets[j][b_good]
                _test_targets = targets[j]
                train_sets_withNone = ori[b_good]
            else:
                if standard[j] >= 0.1:
                    _train_sets = np.vstack((_train_sets, train_sets[j][b_good]))
                    _targets = np.hstack((_targets, targets[j]))
                    train_sets_withNone = np.vstack((train_sets_withNone, ori[b_good]))
                else:
                    _test_sets = np.vstack((_test_sets, train_sets[j][b_good]))
                    _test_targets = np.hstack((_test_targets, targets[j]))
    train_sets = _train_sets.reshape(-1, 4)
    test_sets = _test_sets.reshape(-1, 4)
    train_sets_withNone = train_sets_withNone.reshape(-1, 4)
    targets = _targets
    print('train_sets = ', train_sets)
    print('targets = ', targets)
    print('train_sets.shape = ', train_sets.shape)
    print('targets.shape = ', targets.shape)
    print('test_sets.shape = ', test_sets.shape)
    print('test_targets.shape = ', _test_targets.shape)
    print('train_sets_withNone.shape = ', train_sets_withNone.shape)
    # print(train_sets)
    # print(train_sets_withNone)
    if model_type == 'bp':
        train_set, target = Variable(torch.from_numpy(train_sets).float()), \
                        Variable(torch.from_numpy(targets).float())
        test_sets, test_targets = Variable(torch.from_numpy(test_sets).float()), \
                              Variable(torch.from_numpy(_test_targets).float())
        return train_set, target, train_sets_withNone, test_sets, test_targets
    elif model_type == 'xgboost':
        return np.array(train_sets), np.array(targets), train_sets_withNone, test_sets, _test_targets



def plot_results(loss_train, loss_tests, output, target, train_sets_withNone):
    plt.subplot(311)
    plt.plot(loss_train[200:], label='train loss')
    plt.plot(loss_tests[200:], label='test loss')
    plt.legend()
    plt.subplot(312)
    plt.plot(output.data.numpy(), label='Predicted')
    plt.plot(target.data.numpy(), 'k', label='Ground Truth')
    plt.legend()
    plt.subplot(313)
    plt.plot(np.nanmean(train_sets_withNone, axis=1), label='Original Mean Value')
    plt.plot(target.data.numpy(), 'k', label='Ground Truth')
    plt.legend()
    plt.show()


def plot_results2(output, target, train_sets_withNone):
    # plt.subplot(311)
    # plt.plot(loss_train[200:], label='train loss')
    # plt.plot(loss_tests[200:], label='test loss')
    # plt.legend()
    plt.subplot(211)
    plt.plot(output, label='Predicted')
    plt.plot(target, 'k', label='Ground Truth')
    plt.legend()
    plt.subplot(212)
    plt.plot(np.nanmean(train_sets_withNone, axis=1), label='Original Mean Value')
    plt.plot(target, 'k', label='Ground Truth')
    plt.legend()
    plt.show()


def main(load_data=True, plot_loss=True, model_type='bp'):
    if load_data:
        print('loading ...')
        train_sets, targets = np.load('./packed_trainset-24.npy')
        train_sets = train_sets.transpose()
        # sys.exit()
    else:
        [train_sets, targets] = pack_data(save_data=True)
        print('Ending pack data')
    train_set, target, train_sets_withNone, test_sets, test_targets = remove_allnan_produce_testset(train_sets, targets, model_type)

    if model_type == 'bp':
        model = DnnModel(node_nums=[20, 0])
        print(model)
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.0)
        loss_train = []
        loss_tests = []
        for epoch in range(1000):
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 3000], gamma=0.1)
            model.eval()
            y_pre_test = model(test_sets)
            # print(y_pre_test.reshape(-1))
            loss_test = float(F.mse_loss(y_pre_test.reshape(-1), test_targets).item())

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
            # scheduler.step()
            optimizer.step()
            if epoch % 1 == 0:
                loss_train.append(loss.item())
                loss_tests.append(loss_test)
        print('predicted output = ', output.data.numpy().reshape(-1))
        print('Ground Truth = ', target.data.numpy())
        print('Original Mean Value = ', np.nanmean(train_sets_withNone, axis=1))
        # print('Predicted MSE = ', loss.item())
        print('Predicted MAE = ', np.mean(np.abs((output.reshape(-1) - target).data.numpy())))
        # print('Original_Mean_Value_Method MSE = ', np.nanmean((np.nanmean(train_sets_withNone, axis=1) - targets) ** 2))
        print('Original Mean Value Method MAE = ', np.nanmean(np.abs(np.nanmean(train_sets_withNone, axis=1) - target)))
        if plot_loss:
            plot_results(loss_train, loss_tests, output, target, train_sets_withNone)
    elif model_type == 'xgboost':
        model = XGBoost_model(train_set, target)
        output = model.predict(train_set)
        print('predicted output = ', output.reshape(-1))
        print('Ground Truth = ', target)
        print('Original Mean Value = ', np.nanmean(train_sets_withNone, axis=1))
        # print('Predicted MSE = ', loss.item())
        print('Predicted MAE = ', np.mean(np.abs(output.reshape(-1) - target)))
        # print('Original_Mean_Value_Method MSE = ', np.nanmean((np.nanmean(train_sets_withNone, axis=1) - targets) ** 2))
        print('Original Mean Value Method MAE = ', np.nanmean(np.abs(np.nanmean(train_sets_withNone, axis=1) - target)))
        if plot_loss:
            plot_results2(output, target, train_sets_withNone)




if __name__ == '__main__':
    main(load_data=True, plot_loss=True, model_type='xgboost')
    '''Try Xgboost'''




