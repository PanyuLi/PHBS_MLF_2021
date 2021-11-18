# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from itertools import combinations
import bottleneck as bk
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from keras import Sequential, layers, regularizers, losses, optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataHandler(object):
    """
    test_code 
    A = DataHandler()
    A.trade_day()
    y = A.handle_y_data()
    x = A.handle_x_data()
    """

    def __init__(self):
        self.x_data_all = pd.read_excel('data/MLF_project_dataset.xlsx', skiprows=3, index_col=0)
        self.x_data_all = self.x_data_all.replace({np.nan: 0})
        self.y_data = pd.read_excel('data/HS300_close.xlsx', skiprows=3, index_col=0)

        self.time_period = self.x_data_all.index
        self.effective_time = self.time_period[30:-9]  # 前30天没有数据图片，后10年没有十日收益率

        self.stock_name = list(self.y_data.columns)

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
        np.save(file="data/x_data.npy", arr=self.x_data)
        return self.x_data

    def handle_y_data(self):
        self.y_data = self.y_data.pct_change(10).shift(-9)
        self.y_data = self.y_data.mul(self.trade_day)
        self.y_data = np.array(self.y_data.iloc[30:-9, :])
        self.y_data[self.y_data > 0] = 1
        self.y_data[self.y_data <= 0] = 0
        np.save(file="data/y_data.npy", arr=self.y_data)
        return self.y_data


class FeatureExtract(object):
    """
    data dims: (size, num_feature, time)
    """

    def __init__(self, num_feature):
        self.comb = list(combinations(np.arange(num_feature), 2))
        self.rev_comb = self.reverse_comb()

    def reverse_comb(self):
        new_comb = []
        for c in self.comb:
            new_comb.append((c[1], c[0]))
        return new_comb

    def ts_corr3d(self, data, stride):
        data_comb = data[:, self.comb, :]
        data_rev_comb = data[:, self.rev_comb, :]
        cycle = data.shape[2] - stride + 1
        for c in range(cycle):
            loc = c + np.arange(stride)
            sub = data_comb[:, :, :, loc]
            sub_rev = data_rev_comb[:, :, :, loc]
            spread = sub - sub.mean(axis=3, keepdims=True)
            spread_rev = sub_rev - sub_rev.mean(axis=3, keepdims=True)
            cov = ((spread * spread_rev).sum(axis=3, keepdims=True) / (stride - 1))[:, :, 0]
            std = (np.nanstd(sub, axis=3)[:, :, 0] * np.nanstd(sub, axis=3)[:, :, 1])[:, :, None]
            corr = cov / (std + 1e-9)
            if c == 0:
                corr_matrix = corr
            else:
                corr_matrix = np.concatenate((corr_matrix, corr), axis=2)
        return corr_matrix
        # corr_matrix = tf.cast(tf.convert_to_tensor(corr_matrix), tf.float32)
        # return self.batch_norm(corr_matrix)

    def ts_cov3d(self, data, stride):
        data_comb = data[:, self.comb, :]
        data_rev_comb = data[:, self.rev_comb, :]
        cycle = data.shape[2] - stride + 1
        for c in range(cycle):
            loc = c + np.arange(stride)
            sub = data_comb[:, :, :, loc]
            sub_rev = data_rev_comb[:, :, :, loc]
            spread = sub - sub.mean(axis=3, keepdims=True)
            spread_rev = sub_rev - sub_rev.mean(axis=3, keepdims=True)
            cov = ((spread * spread_rev).sum(axis=3, keepdims=True) / (stride - 1))[:, :, 0]
            if c == 0:
                cov_matrix = cov
            else:
                cov_matrix = np.concatenate((cov_matrix, cov), axis=2)
        return cov_matrix
        # cov_matrix = tf.cast(tf.convert_to_tensor(cov_matrix), tf.float32)
        # return self.batch_norm(cov_matrix)

    def ts_std3d(self, data, stride):
        roll_std = bk.move_std(data, stride)[:, :, stride - 1:]
        return roll_std
        # roll_std = tf.cast(tf.convert_to_tensor(roll_std), tf.float32)
        # return self.batch_norm(roll_std)

    def ts_zscore3d(self, data, stride):
        zscore = (data - bk.move_mean(data, stride)) / (bk.move_std(data, stride) + 1e-9)
        zscore = zscore[:, :, stride - 1:]
        return zscore
        # zscore = tf.cast(tf.convert_to_tensor(zscore), tf.float32)
        # return self.batch_norm(zscore)

    def ts_return3d(self, data, stride):
        # 涨跌幅的变化？,inf?
        cycle = data.shape[2] - stride + 1
        for c in range(cycle):
            sub = data[:, :, c:c + stride]
            ret = (sub[:, :, -1] / (sub[:, :, 0] + 1e-9) - 1)[:, :, None]
            # ret[np.isin(ret, [np.inf, -np.inf])] = 1e6
            if c == 0:
                ret_matrix = ret
            else:
                ret_matrix = np.concatenate((ret_matrix, ret), axis=2)
        return ret_matrix
        # ret_matrix = tf.cast(tf.convert_to_tensor(ret_matrix), tf.float32)
        # return self.batch_norm(ret_matrix)

    def ts_decay3d(self, data, stride):
        cycle = data.shape[2] - stride + 1
        for c in range(cycle):
            sub = data[:, :, c:c + stride]
            weight = np.arange(stride) + 1
            weight = weight / weight.sum()
            wma = (sub * weight).sum(axis=2, keepdims=True)
            if c == 0:
                wma_matrix = wma
            else:
                wma_matrix = np.concatenate((wma_matrix, wma), axis=2)
        return wma_matrix
        # wma_matrix = tf.cast(tf.convert_to_tensor(wma_matrix), tf.float32)
        # return self.batch_norm(wma_matrix)

    def ts_mean3d(self, data, stride):
        roll_mean = bk.move_mean(data, stride)[:, :, stride - 1:]
        return roll_mean
        # roll_mean = tf.cast(tf.convert_to_tensor(roll_mean), tf.float32)
        # return self.batch_norm(roll_mean)

    def ts_max3d(self, data, stride):
        roll_max = bk.move_max(data, stride)[:, :, stride - 1:]
        return roll_max
        # roll_max = tf.cast(tf.convert_to_tensor(roll_max), tf.float32)
        # return roll_max

    def ts_min3d(self, data, stride):
        roll_min = bk.move_min(data, stride)[:, :, stride - 1:]
        return roll_min
        # roll_min = tf.cast(tf.convert_to_tensor(roll_min), tf.float32)
        # return self.batch_norm(roll_min)

    def feature_concat(self, data, stride):
        """
        output format: [N, W, H, C]
        """
        ts_cov = self.ts_cov3d(data, stride)
        ts_corr = self.ts_corr3d(data, stride)
        ts_std = self.ts_std3d(data, stride)
        ts_zscore = self.ts_zscore3d(data, stride)
        ts_decay = self.ts_decay3d(data, stride)
        ts_return = self.ts_return3d(data, stride)
        ts_mean = self.ts_mean3d(data, stride)
        ts_min = self.ts_mean3d(data, stride)
        ts_max = self.ts_mean3d(data, stride)
        ftr_concat = np.concatenate((ts_cov, ts_corr, ts_std, ts_decay, ts_zscore,
                                     ts_return, ts_mean, ts_min, ts_max), axis=1)
        ftr_concat = ftr_concat[:, None, :, :]
        return ftr_concat


class AlphaNet(nn.Module):
    def __init__(self, input_shape, stride_cnn, stride_pool, dropout, flatten_node, fc_neuron):
        super(AlphaNet, self).__init__()
        self.channel = input_shape[0]
        self.width = input_shape[1]  # num features
        self.height = input_shape[2]  # time series
        self.stride_cnn = stride_cnn
        self.stride_pool = stride_pool
        self.fc1_neuron = fc_neuron[0]
        self.fcast_neuron = fc_neuron[1]
        self.batchnorm = nn.BatchNorm2d(self.channel)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(flatten_node, self.fc1_neuron)
        self.out = nn.Linear(self.fc1_neuron, self.fcast_neuron)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.comb = list(combinations(np.arange(self.width), 2))
        self.rev_comb = self.reverse_comb()

    def reverse_comb(self):
        new_comb = []
        for c in self.comb:
            new_comb.append((c[1], c[0]))
        return new_comb

    def forward(self, data):
        """
        data fromat: [N, C, W, H]
        """
        conv1 = self.ts_cov4d(data, self.stride_cnn).to(torch.float)
        bn1 = self.batchnorm(conv1)
        conv2 = self.ts_corr4d(data, self.stride_cnn).to(torch.float)
        bn2 = self.batchnorm(conv2)
        conv3 = self.ts_std4d(data, self.stride_cnn).to(torch.float)
        bn3 = self.batchnorm(conv3)
        conv4 = self.ts_decay4d(data, self.stride_cnn).to(torch.float)
        bn4 = self.batchnorm(conv4)
        conv5 = self.ts_zscore4d(data, self.stride_cnn).to(torch.float)
        bn5 = self.batchnorm(conv5)
        conv6 = self.ts_return4d(data, self.stride_cnn).to(torch.float)
        bn6 = self.batchnorm(conv6)
        conv7 = self.ts_return4d(data, self.stride_cnn).to(torch.float)
        bn7 = self.batchnorm(conv7)
        #         print(conv1.size(),conv2.size(),conv3.size(),conv4.size(),conv5.size(),conv6.size(),conv7.size())
        #         print(bn1.size(),bn2.size(),bn3.size(),bn4.size(),bn5.size(),bn6.size(),bn7.size())
        conv_layer = torch.cat([bn1, bn2, bn3, bn4, bn5, bn6, bn7], axis=2)
        #         print(conv_layer.size())

        ts_max = self.ts_pool4d(conv_layer, self.stride_pool, method='max')
        ts_max = self.batchnorm(ts_max)
        ts_min = self.ts_pool4d(conv_layer, self.stride_pool, method='min')
        ts_min = self.batchnorm(ts_min)
        ts_mean = self.ts_pool4d(conv_layer, self.stride_pool, method='mean')
        ts_mean = self.batchnorm(ts_mean)
        pool_layer = torch.cat([ts_max, ts_min, ts_mean], axis=2)
        #         print(ts_max.size(),ts_min.size(),ts_mean.size(),pool_layer.size())

        flatten_layer = pool_layer.flatten(start_dim=1)
        input_size = flatten_layer.size(1)
        # print(input_size)

        # fc_layer1 = nn.Linear(input_size, self.fc1_neuron)(flatten_layer)
        # activate_layer = nn.ReLU()(fc_layer1)
        # dropout_layer = nn.Dropout(0.5)(activate_layer)
        # out_layer = nn.Linear(self.fc1_neuron,self.fcast_neuron)(dropout_layer)

        fc_layer1 = self.dropout(self.relu(self.fc1(flatten_layer)))
        out_layer = self.sigmoid(self.out(fc_layer1))

        return out_layer.to(torch.float)

    def ts_corr4d(self, data, stride):
        data_comb = data[:, :, self.comb, :]
        data_rev_comb = data[:, :, self.rev_comb, :]
        cycle = self.height - stride + 1
        for c in range(cycle):
            loc = c + np.arange(stride)
            sub = data_comb[:, :, :, :, loc]
            sub_rev = data_rev_comb[:, :, :, :, loc]
            spread = sub - sub.mean(axis=4, keepdims=True)
            spread_rev = sub_rev - sub_rev.mean(axis=4, keepdims=True)
            cov = ((spread * spread_rev).sum(axis=4, keepdims=True) / (stride - 1))[:, :, :, 0]
            std = (np.nanstd(sub, axis=4)[:, :, :, 0] * np.nanstd(sub, axis=4)[:, :, :, 1])[:, :, :, None]
            corr = cov / (std + 1e-9)
            # corr[np.isin(corr, [np.inf, -np.inf])] = 1e6
            if c == 0:
                corr_matrix = corr
            else:
                corr_matrix = np.concatenate((corr_matrix, corr), axis=3)
        return torch.from_numpy(corr_matrix)

    def ts_cov4d(self, data, stride):
        data_comb = data[:, :, self.comb, :]
        data_rev_comb = data[:, :, self.rev_comb, :]
        cycle = self.height - stride + 1
        for c in range(cycle):
            loc = c + np.arange(stride)
            sub = data_comb[:, :, :, :, loc]
            sub_rev = data_rev_comb[:, :, :, :, loc]
            spread = sub - sub.mean(axis=4, keepdims=True)
            spread_rev = sub_rev - sub_rev.mean(axis=4, keepdims=True)
            cov = ((spread * spread_rev).sum(axis=4, keepdims=True) / (stride - 1))[:, :, :, 0]
            if c == 0:
                cov_matrix = cov
            else:
                cov_matrix = np.concatenate((cov_matrix, cov), axis=3)
        return torch.from_numpy(cov_matrix)

    def ts_std4d(self, data, stride):
        roll_std = bk.move_std(data, stride)[:, :, :, stride - 1:]
        return torch.from_numpy(roll_std)

    def ts_zscore4d(self, data, stride):
        zscore = (data - bk.move_mean(data, stride)) / (bk.move_std(data, stride) + 1e-9)
        zscore = zscore[:, :, :, stride - 1:]
        return torch.from_numpy(zscore)

    def ts_return4d(self, data, stride):
        # 涨跌幅的变化？,inf?
        cycle = self.height - stride + 1
        for c in range(cycle):
            sub = data[:, :, :, c:c + stride]
            sub[:, :, :, 0][np.where(sub[:, :, :, 0])]
            ret = (sub[:, :, :, -1] / (sub[:, :, :, 0] - 1) + 1e-9)[:, :, :, None]
            ret[np.isin(ret, [np.inf, -np.inf])] = 1e6
            if c == 0:
                ret_matrix = ret
            else:
                ret_matrix = np.concatenate((ret_matrix, ret), axis=3)
        return torch.from_numpy(ret_matrix)

    def ts_decay4d(self, data, stride):
        cycle = self.height - stride + 1
        for c in range(cycle):
            sub = data[:, :, :, c:c + stride]
            weight = np.arange(stride) + 1
            weight = weight / weight.sum()
            wma = (sub * weight).sum(axis=3, keepdims=True)
            if c == 0:
                wma_matrix = wma
            else:
                wma_matrix = np.concatenate((wma_matrix, wma), axis=3)
        return torch.from_numpy(wma_matrix)

    def ts_mean4d(self, data, stride):
        roll_mean = bk.move_mean(data.detach().numpy(), stride)[:, :, :, stride - 1:]
        return torch.from_numpy(roll_mean)

    def ts_max4d(self, data, stride):
        roll_max = bk.move_max(data.detach().numpy(), stride)[:, :, :, stride - 1:]
        return torch.from_numpy(roll_max)

    def ts_min4d(self, data, stride):
        roll_min = bk.move_min(data.detach().numpy(), stride)[:, :, :, stride - 1:]
        return torch.from_numpy(roll_min)

    def ts_pool4d(self, data, stride, method):
        if type(data) == torch.Tensor:
            data = data.detach().numpy()
        if data.shape[-1] <= stride:
            step_list = [0, data.shape[-1]]
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        feature_num = data.shape[2]
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        global l
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            if method == 'max':
                sub_data1 = data[:, :, :, start:end].max(axis=3, keepdims=True)
            if method == 'min':
                sub_data1 = data[:, :, :, start:end].min(axis=3, keepdims=True)
            if method == 'mean':
                sub_data1 = data[:, :, :, start:end].mean(axis=3, keepdims=True)
            l.append(sub_data1)
        try:
            pool_matrix = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, 1, feature_num, len(step_list) - 1)
        except:
            pool_matrix = np.squeeze(np.array(l)).reshape(-1, 1, feature_num, len(step_list) - 1)
        return torch.from_numpy(pool_matrix)

    def train_net(self, epochs, trainLoader, validLoader, model, Lr, momen):
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters())
        # loss_list = []
        # for e in range(epochs):
        #     train_loss = 0.0
        #     model.train()
        #     for i, (data, labels) in enumerate(trainLoader):
        #         out = model(data.detach().numpy())
        #         # loss = nn.functional.binary_cross_entropy_with_logits(out, labels.to(torch.float))
        #         loss = criterion(out, labels.to(torch.float))

        #         optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
        #         train_loss += loss.item()
        #         loss.backward()
        #         optimizer.step()
        #         loss_list.append(train_loss)
        #     print("Current epoch time: {}, Loss: {}".format(e+1, loss_list[-1]))  
        loss_list = []
        for e in range(epochs):
            train_loss = 0.0
            model.train()
            for data, labels in trainLoader:
                target = model(data.detach().numpy())
                loss = nn.functional.binary_cross_entropy_with_logits(target, labels.to(torch.float))
                # loss = criterion(target, labels.to(torch.float))

                optimizer.zero_grad()  # if don't call zero_grad, the grad of each batch will be accumulated
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                loss_list.append(train_loss)

            valid_loss = 0.0
            model.eval()  # Optional when not using Model Specific layer
            for data, labels in validLoader:
                target = model(data.detach().numpy())
                loss = nn.functional.binary_cross_entropy_with_logits(target, labels.to(torch.float))
                valid_loss += loss.item()

            print('Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(e + 1, train_loss / len(trainLoader),
                                                                             valid_loss / len(validLoader)))
            # print("Current epoch time: {}, Loss: {}".format(e+1, loss_list[-1]))  

    def predict(self, data, model):
        model.eval()
        with torch.no_grad():
            out_probi = model(data).data.numpy()
            # out_value = np.where(out_probi>0.5, 1, 0)
            return out_probi

    def evaluate(self, test_x, test_y, model):
        # when in test stage, no grad
        model.eval()
        with torch.no_grad():
            predict_y = self.predict(test_x, model)
            predict_y = np.where(predict_y > 0.5, 1, 0)
            correct = (predict_y == test_y).sum()
            print('Accuracy: {}'.format(correct / len(test_y)))


class ModifyDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class CNNnet(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape  # channel_last

    def forward(self, nodes=16):
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())

        model.add(layers.Dense(nodes, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        # model.compile(optimizer='rmsprop', loss=losses.mean_squared_error, metrics=['accuracy'])
        model.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def fit_net(self, model, train_x, train_y, test_x, test_y, epochs=10, batch_size=1000):
        train_x = np.squeeze(train_x)[:, :, :, None]
        test_x = np.squeeze(test_x)[:, :, :, None]
        history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y))
        return history

    def predict_net(self, model, test_x):
        test_x = np.squeeze(test_x)[:, :, :, None]
        predictions = model.predict(test_x)
        # predictions = np.where(predictions>0.5, 1, 0)
        return predictions

    def evaluate_net(self, model, test_x, test_y):
        test_x = np.squeeze(test_x)[:, :, :, None]
        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
        print("\nTest loss: {}, Test accuracy: {}".format(test_loss, test_acc))
        return test_acc

    def plot(self, history):
        # 绘制精确度曲线
        plt.figure()
        plt.subplot(211)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc='upper right')  # 显示图例

        # 画出损失曲线
        plt.subplot(212)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')  # 显示图例
        plt.show()
