import pandas as pd
import numpy as np
from itertools import combinations
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')



class DataHandler(object):
    """
    test_code
    A = DataHandler()
    A.trade_day()
    y = A.handle_y_data()
    x = A.handle_x_data()
    """

    def __init__(self):
        self.x_data_all = pd.read_excel('../data/MLF_project_dataset.xlsx', skiprows=3, index_col=0)
        self.x_data_all = self.x_data_all.replace({np.nan: 0})
        self.y_data = pd.read_excel('../data/HS300_close.xlsx', skiprows=3, index_col=0)

        self.time_period = self.x_data_all.index
        self.effective_time = self.time_period[30:-9]  # 前30天没有数据图片，后10年没有十日收益率

        self.stock_name = list(self.y_data.columns)

    def handle_y_data(self):
        self.y_data = self.y_data.pct_change(10).shift(-9)
        self.y_data = self.y_data.mul(self.trade_day)
        self.y_data = np.array(self.y_data.iloc[30:-9, :])
        self.y_data[self.y_data > 0] = 1
        self.y_data[self.y_data <= 0] = 0
        np.save(file="../data/y_data.npy", arr=self.y_data)
        return self.y_data

    # 判断是否停牌
    def trade_day(self):
        self.trade_day = self.x_data_all.iloc[:, range(4, self.x_data_all.shape[1], 11)]
        self.trade_day = self.trade_day.replace({0: np.nan})
        self.trade_day = (self.trade_day.notnull() + 0).replace({0: np.nan})
        self.trade_day.columns = self.stock_name

    def handle_x_data(self):
        self.x_data = np.zeros(((len(self.time_period) - 39), len(self.stock_name), 9, 30), dtype=float)
        for i in range(0, len(self.stock_name)):
            for j in range(0, (len(self.time_period) - 39)):
                if self.trade_day.iloc[j + 30, i] == 1:
                    data = np.array(self.x_data_all.iloc[j:j + 30, (11 * i):(11 * i + 9)])
                    self.x_data[j, i] = data.T
                else:
                    self.x_data[j, i] = np.nan
        np.save(file="../data/x_data.npy", arr=self.x_data)
        return self.x_data

    def handle_single_x(self):
        self.x_sing_data = np.zeros(((len(self.time_period) - 39), len(self.stock_name), 9, 1),dtype=float)
        for i in range(0, len(self.stock_name)):
            for j in range(0,  (len(self.time_period) - 39)):
                if self.trade_day.iloc[j+30, i] == 1:
                    data = np.array(self.x_data_all.iloc[j:j+1, (11*i):(11*i+9)])
                    self.x_sing_data[j,i] = data.T
                else:
                    self.x_sing_data[j,i] = np.nan
        np.save(file="../data/x_sing_data.npy", arr=self.x_sing_data)
        return self.x_sing_data

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

    np.save(file='../init/init_test_y_{}.npy'.format(str(batch_code)), arr=init_test_y)
    np.save(file='../loc/loc_y_{}.npy'.format(str(batch_code)), arr=np.array(loc_list))
    train_x = (train_x - np.mean(train_x, axis=1, keepdims=True)) / np.std(train_x, axis=1, keepdims=True)
    test_x = (test_x - np.mean(test_x, axis=1, keepdims=True)) / np.std(test_x, axis=1, keepdims=True)

    return train_x[:, None, :, :], train_y[:, None], test_x[:, None, :, :], test_y[:, None]

def clear_nan(array_x, array_y):
    na_listx = []
    for i in range(len(array_x)):
        if np.isnan(array_x[i]).any():
            na_listx.append(i)

    na_listy = []
    for i in range(len(array_y)):
        if np.isnan(array_y[i]).any():
            na_listy.append(i)

    total_na_list = list(set(na_listx) | set(na_listy))
    x_index_list = list(set(np.arange(len(array_x))) - set(total_na_list))
    y_index_list = list(set(np.arange(len(array_y))) - set(total_na_list))
    array_x = array_x[x_index_list]
    array_y = array_y[y_index_list]
    return array_x, array_y


def build_xgboost(train_x, test_x, train_y, test_y):
    train_x = np.squeeze(train_x.sum(axis=-1))
    test_x = np.squeeze(test_x.sum(axis=-1))
    train_y, test_y = train_y.flatten().astype(int), test_y.flatten().astype(int)
    train_x, train_y = clear_nan(train_x, train_y)
    test_x, test_y = clear_nan(test_x, test_y)


    model = make_pipeline(StandardScaler(), PCA(n_components=4), XGBClassifier(probability=True))
    param_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear', 'rbf']}]

    param_test5 = {
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],  # 0.005
        'max_depth':[3,4,5,6]
    }

    grid = GridSearchCV(estimator=XGBClassifier(
        learning_rate=0.1,  # 学习率，控制每次迭代更新权重时的步长，默认0.3;调参：值越小，训练越慢;典型值为0.01-0.2
        n_estimators=50,  # 总共迭代的次数，即决策树的个数
        min_child_weight=5,  # 叶子节点最小权重;默认值为1;调参：值越大，越容易欠拟合；值越小，越容易过拟合
        gamma=0.0,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值
        subsample=0.6,  # 训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1;调参：防止overfitting
        colsample_bytree=0.7,  # 随机选择N%特征建立决策树;防止overfitting
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        # 解决样本个数不平衡的问题;正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10.
        seed=27),
        param_grid=param_test5,
        scoring='roc_auc',
        n_jobs=4, cv=5)

    gs = grid.fit(train_x, train_y)

    print(gs.best_score_)
    print('---------')
    print(gs.best_params_)
    print('---------')
    print(gs.cv_results_)
    clf = gs.best_estimator_
    predict_y = clf.predict(test_x)
    predict_prob = clf.predict_proba(test_x)[:, 1]
    return clf, predict_prob, predict_y


def build_xgboost_v2(train_x, test_x, train_y, test_y):
    train_x = np.squeeze(train_x.sum(axis=-1))
    test_x = np.squeeze(test_x.sum(axis=-1))
    train_y, test_y = train_y.flatten().astype(int), test_y.flatten().astype(int)
    train_x, train_y = clear_nan(train_x, train_y)
    test_x, test_y = clear_nan(test_x, test_y)

    xgb = XGBClassifier(
        learning_rate=0.1,  # 学习率，控制每次迭代更新权重时的步长，默认0.3;调参：值越小，训练越慢;典型值为0.01-0.2
        n_estimators=50,  # 总共迭代的次数，即决策树的个数
        min_child_weight=5,  # 叶子节点最小权重;默认值为1;调参：值越大，越容易欠拟合；值越小，越容易过拟合
        gamma=0.0,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值
        subsample=0.6,  # 训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1;调参：防止overfitting
        colsample_bytree=0.7,  # 随机选择N%特征建立决策树;防止overfitting
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        # 解决样本个数不平衡的问题;正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10.
        seed=27)

    model = make_pipeline(StandardScaler(), PCA(n_components=4), xgb)
    #param_range = [0.01, 0.1, 1.0, 10.0, 100.0]


    param_test5 = {
        'xgbclassifier__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
        'xgbclassifier__max_depth':[3,4,5,6]
    }

    grid = GridSearchCV(estimator=model,
        param_grid=param_test5,
        scoring='roc_auc',
        n_jobs=4, cv=5)

    gs = grid.fit(train_x, train_y)

    print(gs.best_score_)
    print('---------')
    print(gs.best_params_)
    print('---------')
    print(gs.cv_results_)
    clf = gs.best_estimator_
    predict_y = clf.predict(test_x)
    predict_prob = clf.predict_proba(test_x)[:, 1]
    return clf, predict_prob, predict_y




y = np.load('../data/y_data.npy').astype('float')
x = np.load('../data/x_sing_data.npy').astype('float')
train_size = 500
num_batch = int((len(x) - train_size) / 120)
data_x, data_y, test_x, test_y = drop_nan_data(x, y, 14, train_size, 5)
print(data_x.shape)
print(data_y.shape)
print(test_x.shape)
print(test_y.shape)
print(x.shape)
print(y.shape)
print(data_y.flatten().astype(int).shape)

#xx = np.load('../data/x_data.npy').astype('float')
#data_xx, data_yy, test_xx, test_yy = drop_nan_data(xx, y, 14, train_size, 5)


a,b,c = build_xgboost_v2(data_x, test_x, data_y, test_y)
#x1, y1 = clear_nan(test_x, test_y)
print(c)
#print(r2_score(c, y1))




