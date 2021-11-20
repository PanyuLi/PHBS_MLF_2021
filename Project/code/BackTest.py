# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
# from itertools import combinations
# import bottleneck as bn
import matplotlib.pyplot as plt


class DataHandler(object):
    
    # test_code ########################################
    # A = DataHandler()
    # A.trade_day()
    # y = A.handle_y_data()
    # x = A.handle_x_data()
    
    def __init__(self):
        self.x_data_all = pd.read_excel('MLF_project_dataset.xlsx', skiprows=3, index_col=0)
        self.x_data_all = self.x_data_all.replace({np.nan:0})
        self.y_data = pd.read_excel('HS300_close.xlsx', skiprows=3, index_col=0)
        
        self.time_period = self.x_data_all.index
        self.effective_time = self.time_period[30:-9] # 前30天没有数据图片，后10年没有十日收益率
        
        self.stock_name = list(self.y_data.columns)       
             
    #判断是否停牌
    def trade_day(self):    
        self.trade_day = self.x_data_all.iloc[:,range(4,self.x_data_all.shape[1],11)]
        self.trade_day = self.trade_day.replace({0:np.nan})
        self.trade_day =  (self.trade_day.notnull()+0).replace({0:np.nan})
        self.trade_day.columns = self.stock_name
        self.trade_day.to_excel('trade_day.xlsx')

    def handle_x_data(self):
        self.x_data = np.zeros(((len(self.time_period)-39),len(self.stock_name),9,30),dtype = float)
        for i in range(0, len(self.stock_name)):
            for j in range(0, (len(self.time_period)-39)):
                if self.trade_day.iloc[j+30,i]==1:                    
                    data = np.array(self.x_data_all.iloc[j:j+30,(11*i):(11*i+9)])
                    self.x_data[j,i] = data.T
                    # if np.isnan(data).any()==True:
                    #     self.x_data[j,i] = np.nan
                else:
                    self.x_data[j,i] = np.nan 
        np.save(file="x_data.npy", arr=self.x_data)
        return self.x_data
            
    def handle_y_data(self):
        day10_return = self.y_data.pct_change(10).shift(-9)
        day10_return = day10_return.mul(self.trade_day)
        day10_return  = np.array(day10_return.iloc[30:-9,:])
        np.save(file="y_data.npy", arr=day10_return )
        return day10_return 
    
    def handel_daily_return(self):
        daily_return = self.y_data.pct_change(1)
        daily_return = daily_return.mul(self.trade_day)
        daily_return  = np.array(daily_return.iloc[30:-9,:])
        np.save(file="daily_return.npy", arr=daily_return)
        return daily_return 


# A = DataHandler()
# trade_day = A.trade_day()
# y = A.handle_y_data()
# x = A.handle_x_data()
# daily_return = A.handel_daily_return()


# b0 = np.load(file="batch_predict_0.npy")
# b1 = np.load(file="batch_predict_1.npy")
# b2 = np.load(file="batch_predict_2.npy")
# b3 = np.load(file="batch_predict_3.npy")
# b4 = np.load(file="batch_predict_4.npy")
# b = np.vstack((b0, b1, b2, b3, b4))

# yy0 = np.load(file="loc_y_0.npy")
# yy1 = np.load(file="loc_y_1.npy")
# yy2 = np.load(file="loc_y_2.npy")
# yy3 = np.load(file="loc_y_3.npy")
# yy4 = np.load(file="loc_y_4.npy")
# yy = np.vstack((yy0, yy1, yy2, yy3, yy4))

# y0 = np.load(file="init_test_y_0.npy")
# y1 = np.load(file="init_test_y_1.npy")
# y2 = np.load(file="init_test_y_2.npy")
# y3 = np.load(file="init_test_y_3.npy")
# y4 = np.load(file="init_test_y_4.npy")
# y = np.vstack((y0, y1, y2, y3, y4))

# ln_mv = pd.read_excel('val_lnmv.xlsx', skiprows=3, index_col=0)


# 输入格式全部为dataframe
class BackTest(object):
    def __init__(self, position, daily_return, window=10, mv_weighted=False, ln_mv=None):
        self.window = window
        
        self.position = pd.DataFrame(np.repeat(np.array(position), self.window, axis=0))
        if mv_weighted:
            self.ln_mv = ln_mv.reset_index(drop=True)
            self.ln_mv.columns = range(len(self.ln_mv.columns))
            weight = self.ln_mv.div(self.ln_mv.sum(axis=1),axis=0)
        else:
            weight = self.position.div(self.position.sum(axis=1),axis=0)
        self.position = self.position.mul(weight)
        
        self.daily_return = daily_return.reset_index(drop=True)
        self.daily_return.columns = range(len(self.daily_return.columns))
        
      # 计算收益率
    def daily_ret(self):
        daily_return = self.position.mul(self.daily_return).sum(axis=1)
        return daily_return
    
    # 计算净值曲线
    def net_value(self, daily_return):
        net_value = (1+daily_return).cumprod() # 日收益率和仓位相乘后累计连乘
        return net_value

    # 计算最大回撤 1-当天净值/累积最大净值
    def drawdown(self, net_value):
        drawdown = 1 - net_value.div(net_value.cummax())
        return drawdown.iloc[-1]

    # 计算夏普比
    def sharpe_ratio(self, net_value, daily_return):
        yearly_return = net_value.iloc[-1]**(250/len(net_value)) - 1
        yearly_volatility = daily_return.std() * np.sqrt(250)
        sharp_ratio = yearly_return/yearly_volatility
        return [yearly_return, yearly_volatility, sharp_ratio]
    
    def output_result(self, plot=True):
        daily_return = self.daily_ret()
        net_value = self.net_value(daily_return)
        result = self.sharpe_ratio(net_value, daily_return)
        self.results = {"年化收益率": result[0], "年化波动率": result[1],
                        "夏普比率": result[2], "最大回撤":self.drawdown(net_value)}
        if plot:
            IF300 = pd.read_excel('IF300.xlsx', skiprows=3, index_col=0)
            IF300 = IF300.pct_change()
            IF300 = IF300.iloc[530:2570,:]
            IF300 = IF300.add(1).cumprod()
            
            plt.figure(figsize=(12, 8))
            plt.plot(IF300.reset_index(drop=True), color='r',label = 'IF300')
            plt.plot(net_value, color='b',label = 'Synthetic factors')
            plt.legend()
            plt.show()
        return self.results

#所用数据格式均为pd.DataFrame
class SingleFactorValidityTest(object):
    def __init__(self, factor, sto_ret_10day, mv_neutral=False, mv_weighted=False, ln_mv=None):
        self.factor = pd.DataFrame(factor)
        self.ln_mv = ln_mv
        self.mv_neutral = mv_neutral
        self.mv_weighted = mv_weighted
        
        if self.mv_neutral:
            self.ln_mv = self.ln_mv.reset_index(drop=True)
            self.ln_mv.columns = range(len(self.ln_mv.columns))
            a = self.ln_mv.sub(self.ln_mv.mean(axis=1),axis=0)
            b = self.factor.sub(self.factor.mean(axis=1),axis=0)
            beta_1 = (a*b).sum(axis=1)/(a**2).sum(axis=1)
            beta_0 = self.factor.mean(axis=1) - beta_1 * self.ln_mv.mean(axis=1)
            residual = self.factor.sub(self.ln_mv.mul(beta_1, axis=0).add(beta_0, axis=0),axis=0)
            self.factor = residual
            
        self.sto_ret_10day = sto_ret_10day.reset_index(drop=True)
        self.sto_ret_10day.columns = range(len(self.sto_ret_10day.columns))

    def cum_IC(self, fac, stock_ret): 
        factor_mean = fac.mean(axis=1)
        strock_return_mean = stock_ret.mean(axis=1)
        X = fac.sub(factor_mean, axis=0)
        Y = stock_ret.sub(strock_return_mean,axis=0)
        IC = X.mul(Y).sum(axis=1) / (X.mul(X).sum(axis=1) * Y.mul(Y).sum(axis=1))**0.5
        cum_IC = IC.cumsum()
        
        return (IC, cum_IC)
    
    def cum_Rank_IC(self):
        factor = self.factor.rank(axis=1)
        stock_return = self.sto_ret_10day.rank(axis=1)
        Rank_IC, cum_Rank_IC = self.cum_IC(factor, stock_return)
        
        return (Rank_IC, cum_Rank_IC)
    
    def IR(self, IC):
        return IC.mean()/IC.std()
     
    def plot_cum_IC(self, cum_IC, title):
        plt.title(title)
        plt.plot(cum_IC)
        plt.show()
        
    def out_put(self):
        IC, cum_IC = self.cum_IC(self.factor, self.sto_ret_10day)
        Rank_IC, cum_Rank_IC = self.cum_Rank_IC()
        IR =self.IR(IC)
        Rank_IR = self.IR(Rank_IC)
        result = {"IC_mean":IC.mean(), "Rank_IC_mean":Rank_IC.mean(), 
                  "IR":IR, "Rank_IR":Rank_IR, 
                  "IC":IC, "cum_IC":cum_IC, 
                  "Rank_IC":Rank_IC, "cum_Rank_IC":cum_Rank_IC}
        
        # self.plot_cum_IC(cum_IC, "cum_IC")
        # self.plot_cum_IC(cum_Rank_IC, "cum_Rank_IC")
        
        return result
    
    def HierarchicalBacktest(self): 
        self.return_5layer = pd.DataFrame(np.nan,index=self.factor.index,columns=[1,2,3,4,5])
        
        self.position = pd.DataFrame(1,index=self.factor.index,columns=self.factor.columns)
        
        factor_quantile = self.factor.quantile(q=[0.2,0.4,0.6,0.8], axis=1)
        
        if self.mv_weighted:
            self.ln_mv = self.ln_mv.reset_index(drop=True)
            self.ln_mv.columns = range(len(self.ln_mv.columns))
        
        for i in self.factor.index:
            layer1 = self.factor.loc[i,:][self.factor.loc[i,:]<factor_quantile.loc[0.2,i]]
            layer2 = self.factor.loc[i,:][(self.factor.loc[i,:]<factor_quantile.loc[0.4,i])&(self.factor.loc[i,:]>=factor_quantile.loc[0.2,i])]
            layer3 = self.factor.loc[i,:][(self.factor.loc[i,:]<factor_quantile.loc[0.6,i])&(self.factor.loc[i,:]>=factor_quantile.loc[0.4,i])]
            layer4 = self.factor.loc[i,:][(self.factor.loc[i,:]<factor_quantile.loc[0.8,i])&(self.factor.loc[i,:]>=factor_quantile.loc[0.6,i])]
            layer5 = self.factor.loc[i,:][self.factor.loc[i,:]>=factor_quantile.loc[0.8,i]]
            layer = [layer1, layer2, layer3, layer4, layer5]
            
            for j in [0,1,2,3,4]:
                sto_ret = self.sto_ret_10day.loc[i,layer[j].index]
                position = self.position.loc[i,layer[j].index]
                if self.mv_weighted:
                    weight = self.ln_mv.loc[i,layer[j].index]/self.ln_mv.loc[i,layer[j].index].sum()
                else:
                    weight = self.position.loc[i,layer[j].index]/self.position.loc[i,layer[j].index].sum()
                self.return_5layer.loc[i,(j+1)] = (sto_ret * position * weight).sum()
            
        self.return_5layer = 1 + self.return_5layer
        self.return_5layer = self.return_5layer.cumprod()
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.return_5layer[1],label = 'Group1')
        plt.plot(self.return_5layer[2],label = 'Group2')
        plt.plot(self.return_5layer[3],label = 'Group3')
        plt.plot(self.return_5layer[4],label = 'Group4')
        plt.plot(self.return_5layer[5],label = 'Group5')
        plt.legend()
        plt.show()
        
        
            

        

    
    
    
        
        
    








