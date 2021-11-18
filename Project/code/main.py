# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:53:47 2021

@author: cherry
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from StockPredict import AlphaNet, CNNnet, ModifyDataset, FeatureExtract
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def drop_nan_data(x, y, batch_code, train_size, sampling):
    init_train_x = x[120 * batch_code: 120 * batch_code + train_size]
    init_train_y = y[120 * batch_code: 120 * batch_code + train_size]
    init_test_x = x[np.arange(120 * batch_code + train_size, 120 * batch_code + train_size + 120, 10)]
    init_test_y = y[np.arange(120 * batch_code + train_size, 120 * batch_code + train_size + 120, 10)]

    if 120 * batch_code + train_size > len(x):
        raise Exception('Out of index error!')

    for i in tqdm(range(0, len(init_train_y), sampling)):
        condi = (~np.isnan(init_train_y[i])) & np.squeeze(~np.isnan(init_train_x[i]).any(axis=-1).any(axis=-1))
        if i == 0:
            train_y = init_train_y[i][condi]
            train_x = init_train_x[i][condi]
        else:
            train_y = np.append(train_y, init_train_y[i][condi])
            train_x = np.concatenate((train_x, init_train_x[i][condi]))

    loc_list = []
    for i in tqdm(range(len(init_test_y))):
        condi = (~np.isnan(init_test_y[i])) & np.squeeze(~np.isnan(init_test_x[i]).any(axis=-1).any(axis=-1))
        if i == 0:
            test_y = init_test_y[i][condi]
            test_x = init_test_x[i][condi]
        else:
            test_y = np.append(test_y, init_test_y[i][condi])
            test_x = np.concatenate((test_x, init_test_x[i][condi]))
        loc_list.append(condi.astype(int))

    np.save(file='init/init_test_y_{}.npy'.format(str(batch_code)), arr=init_test_y)
    np.save(file='loc/loc_y_{}.npy'.format(str(batch_code)), arr=np.array(loc_list))
    train_x = (train_x - np.mean(train_x, axis=1, keepdims=True)) / np.std(train_x, axis=1, keepdims=True)
    test_x = (test_x - np.mean(test_x, axis=1, keepdims=True)) / np.std(test_x, axis=1, keepdims=True)

    return train_x[:, None, :, :], train_y[:, None], test_x[:, None, :, :], test_y[:, None]


def build_alphanet(train_x, val_x, test_x, train_y, val_y, test_y):
    trainset = ModifyDataset(train_x, train_y)
    validset = ModifyDataset(val_x, val_y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True)
    validloader = DataLoader(validset, batch_size=200, shuffle=True)
    alphanet = AlphaNet(input_shape=[1, 9, 30], stride_cnn=10, stride_pool=3, dropout=0.2, flatten_node=2457,
                        fc_neuron=[30, 1])
    alphanet.train_net(epochs=10, trainLoader=trainloader, validLoader=validloader, model=alphanet, Lr=0.001, momen=0)
    predict_y = alphanet.predict(test_x, alphanet)
    alphanet.evaluate(test_x, test_y, alphanet)
    return alphanet, predict_y, None


def build_cnn(train_x, val_x, test_x, train_y, val_y, test_y):
    alphanet = CNNnet(input_shape=(135, 21, 1))
    search = False
    final_node = 16
    if search:
        nodes_param = [16, 32]
        acc = []
        for n in nodes_param:
            print(n)
            model = alphanet.forward(n)
            history = alphanet.fit_net(model, train_x, train_y, val_x, val_y)
            acc.append(alphanet.evaluate_net(model, val_x, val_y))
        final_node = nodes_param[np.argmax(np.array(acc))]
        print(acc, final_node)
    model = alphanet.forward(final_node)
    history = alphanet.fit_net(model, train_x, train_y, val_x, val_y)
    alphanet.evaluate_net(model, test_x, test_y)
    predict_y = alphanet.predict_net(model, test_x)
    return alphanet, predict_y, history


def build_logisticregression(train_x, test_x, train_y, test_y):
    train_x = np.squeeze(train_x.sum(axis=-1))
    test_x = np.squeeze(test_x.sum(axis=-1))
    train_y, test_y = train_y.flatten().astype(int), test_y.flatten().astype(int)

    model = make_pipeline(StandardScaler(), PCA(n_components=3), LogisticRegression())
    param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

    param_grid = [{'logisticregression__C': param_range}]

    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=5,
                      n_jobs=-1)
    gs = gs.fit(train_x, train_y)
    print(gs.best_score_)
    print(gs.best_params_)
    print(gs.cv_results_)
    clf = gs.best_estimator_
    predict_y = clf.predict(test_x)
    predict_prob = clf.predict_proba(test_x)[:, 1]
    return clf, predict_prob, predict_y


def build_svm(train_x, test_x, train_y, test_y):
    train_x = np.squeeze(train_x.sum(axis=-1))
    test_x = np.squeeze(test_x.sum(axis=-1))
    train_y, test_y = train_y.flatten().astype(int), test_y.flatten().astype(int)

    model = make_pipeline(StandardScaler(), PCA(n_components=3), SVC(probability=True))
    # print(pipe_svm.get_params().keys())
    param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear', 'rbf']}]

    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=5,
                      n_jobs=-1)
    gs = gs.fit(train_x, train_y)
    print(gs.best_score_)
    print(gs.best_params_)
    print(gs.cv_results_)
    clf = gs.best_estimator_
    predict_y = clf.predict(test_x)
    predict_prob = clf.predict_proba(test_x)[:, 1]
    return clf, predict_prob, predict_y


def build_randomforest(train_x, test_x, train_y, test_y):
    train_x = np.squeeze(train_x.sum(axis=-1))
    test_x = np.squeeze(test_x.sum(axis=-1))
    train_y, test_y = train_y.flatten().astype(int), test_y.flatten().astype(int)

    pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier())
    # print(pipe_rf.get_params().keys())
    max_depth = list(range(10, 50, 10))
    max_features = list(range(5, 60, 10))
    param_grid = [{'randomforestclassifier__max_depth': max_depth,
                   'randomforestclassifier__max_features': max_features}]

    gs = GridSearchCV(estimator=pipe_rf,
                      param_grid=param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=5,
                      n_jobs=-1)
    gs = gs.fit(train_x, train_y)
    print(gs.best_score_)
    print(gs.best_params_)
    print(gs.cv_results_)
    clf = gs.best_estimator_
    predict_y = clf.predict(test_x)
    predict_prob = clf.predict_proba(test_x)[:, 1]
    return clf, predict_prob, predict_y


def network(data_x, train_x, val_x, test_x, data_y, train_y, val_y, test_y, method):
    if method == 'alphanet':
        netwk, predict_y, _ = build_alphanet(train_x, val_x, test_x, train_y, val_y, test_y)
    if method == 'cnn':
        netwk, predict_y, _ = build_cnn(train_x, val_x, test_x, train_y, val_y, test_y)
    if method == 'lr':
        netwk, predict_y, _ = build_logisticregression(data_x, test_x, data_y, test_y)
    if method == 'svm':
        netwk, predict_y, _ = build_svm(data_x, test_x, data_y, test_y)
    if method == 'randomforest':
        netwk, predict_y, _ = build_randomforest(data_x, test_x, data_y, test_y)
    return netwk, predict_y


if __name__ == '__main__':
    datahandler = DataHandler()
    datahandler.trade_day()
    y = datahandler.handle_y_data()
    x = datahandler.handle_x_data()

    train_size = 500
    num_batch = int((len(x) - train_size) / 120)
    method = ['alphanet', 'cnn', 'lr', 'randomforest']

    predict_res = []
    for m in method:
        for i in range(14, 15):
            data_x, data_y, test_x, test_y = drop_nan_data(x, y, i, train_size, 5)
            print(np.isnan(data_x).any(), np.isnan(data_y).any(), np.isnan(test_x).any(), np.isnan(test_y).any())
            if m != 'alphanet':
                feature_extract = FeatureExtract(9)
                data_x = feature_extract.feature_concat(np.squeeze(data_x), 10)
                test_x = feature_extract.feature_concat(np.squeeze(test_x), 10)
            train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)
            netwk, predict_y = network(data_x, train_x, val_x, test_x, data_y, train_y, val_y, test_y, method=m)
            # predict_res.append(predict_y)

        pred_y = np.where(predict_y > 0.5, 1, 0)
        conf_mat = confusion_matrix(test_y, pred_y)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
        plt.xticks([0, 1], ['P*', 'N*'])
        plt.yticks([0, 1], ['P', 'N'])

        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        plt.tight_layout()
        plt.savefig(f'con_matrix_{m}.png', dpi=300)

        loc_arr = np.load('loc/loc_y_{}.npy'.format(i)).astype(float)
        loc_arr_count = loc_arr.sum(axis=1)
        loc_arr[loc_arr == 0] = np.nan
        left = 0
        for j in range(len(loc_arr_count)):
            right = left + loc_arr_count[j]
            loc_arr[j][loc_arr[j] == 1] = np.squeeze(predict_y[int(left): int(right)])
            left += loc_arr_count[j]
        np.save('predict/{}_predict_w_{}'.format(m, i), arr=loc_arr)
