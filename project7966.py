#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:51:26 2020

@author: shousakai
"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from datetime import datetime
import dateutil
from scipy.stats import norm
from scipy.stats import multivariate_normal
from project796 import *

class Gaussian_Copula:
    def __init__(self, residuals):
        self.data = residuals
        self.mean = residuals.mean()
        self.cov = residuals.cov()
        self.rv = multivariate_normal(self.mean,self.cov)
    def pdf(self,x):
        return (self.rv.pdf(x))
    def cdf(self,x):
        return (self.rv.cdf(x))
    def portfolio_pdf(self,weights):
        port_mean = np.dot(weights, self.mean)
        port_var = np.dot( weights ,  np.dot(self.cov, weights.T) )
        VaR = norm.isf(0.95)*np.sqrt(port_var) + port_mean
        return VaR

Gamma = 6

def objective(weights,cov,mean):
    port_mean = np.dot(weights, mean)
    port_cov = np.dot( weights ,  np.dot(cov,weights.T) )
    obj = port_mean - 0.5*Gamma*port_cov
    return obj
    
def Gradient(weights,cov,mean):
    obj0 = objective(weights,cov,mean)
    d = 0.001
    obj1 = objective( (weights + np.array([d,-d/4,-d/4,-d/4,-d/4]) ),cov,mean)
    obj2 = objective( (weights + np.array([-d/4,d,-d/4,-d/4,-d/4]) ),cov,mean)
    obj3 = objective( (weights + np.array([-d/4,-d/4,d,-d/4,-d/4]) ),cov,mean)
    obj4 = objective( (weights + np.array([-d/4,-d/4,-d/4,d,-d/4]) ),cov,mean)
    obj5 = objective( (weights + np.array([-d/4,-d/4,-d/4,-d/4,d]) ),cov,mean)
    Gradient = np.array([(obj1-obj0)/d, (obj2-obj0)/d, (obj3-obj0)/d, (obj4-obj0)/d, (obj5-obj0)/d,])
    return Gradient

    
def Gradient_Descend(weights,cov,mean,times):
    learning_rate = 0.001
    obj_list = []
    for i in range(times):
        weights -= learning_rate * Gradient(weights,cov,mean)
        obj_list += [objective(weights,cov,mean)]
    return (weights,obj_list)
    
def Nelder_Mead(cov,mean):
    w_list = []
    obj_list = []
    diff = 0.02
    for w1 in np.arange(0,1,diff):
        for w2 in np.arange(0,1-w1,diff):
            for w3 in np.arange(0,1-w1-w2,diff):
                for w4 in np.arange(0,1-w1-w2-w3,diff):
                    w5 = 1 - w1 - w2 - w3 - w4
                    weights = np.array([w1,w2,w3,w4,w5])
                    w_list += [(w1,w2,w3,w4,w5)]
                    obj_list += [objective(weights,cov,mean)]
    return (w_list,obj_list)

def Generate_Weights(mean,cov):
    
    w = cp.Variable(5)
    gamma = cp.Parameter(nonneg=True)
    ret = (np.array(mean))*w 
    risk = cp.quad_form(w, np.array(cov))
    prob = cp.Problem(cp.Maximize(ret - gamma*risk), 
               [cp.sum(w) == 1, 
                w >= 0, w<=1])
    
    gamma.value = 6
    prob.solve()
    return w.value
    
    
    
     
     

if __name__ == '__main__':
    
    tickers = 'AAPL', 'AMZN', 'NKE', 'MSFT', 'GOOG'
    symbols = ['AAPL', 'AMZN', 'NKE', 'MSFT', 'GOOG']
    Data = yf.download(tickers, '2010-01-01', '2020-02-01')
    Data = Data['Adj Close']
    Data.to_csv('Stock.csv')
    
    
    
    data = pd.read_csv("Stock.csv", index_col=0)
    data = data.set_index(pd.to_datetime(data.index))
    logrets = np.log(data/data.shift(1)).dropna()  
    data = logrets
    
    
    
    
    
    
    
    cov = data.cov()
    mean = data.mean()
    rv = multivariate_normal(mean,cov)
    
    weights = np.array([0.5,0.5])
    weights = np.array([0.2,0.2,0.2,0.2,0.2])
    port_mean = np.dot(weights, mean)
    port_cov = np.dot( weights ,  np.dot(cov,weights.T) )
    
    #Out of Sample Test 
    start_date = pd.datetime(2019,1,1)
    end_date = pd.datetime(2019,2,1)
    Time_delta = dateutil.relativedelta.relativedelta(months=1)
    
    Out_of_sample_Returns = np.array([])
    order = np.array([[3, 2], [4, 2], [3, 2], [1, 1], [1, 1]])
    # dcc_all = list(0 for i in range(12))
    # adcc_all = list(0 for i in range(12))
    # dcc_all = pd.read_csv('dcc.csv')
    # dcc_all = np.array(dcc_all)
    # mlp_mean_all = pd.read_csv('mlp_mean.csv')
    # mlp_mean_all = np.array(mlp_mean_all.iloc[:,1])
    # dcc = 
    # mlp_mean_all = pd.read_csv('mlp_mean_all.csv')
    # mlp_mean_all = mlp_mean_all.iloc[:,1:]
    # mlp_mean_all = np.array(mlp_mean_all)
    
    rnn_mean_all = list(0 for i in range(12))
    # rnn_mean_all=pd.read_csv('rnn_mean.csv')
    # rnn_mean_all=rnn_mean_all.iloc[:,1:]
    # rnn_mean_all = np.array(rnn_mean_all)
    # adcc = list(0 for i in range(12))
    # psn_mean_all = list(0 for i in range(12))
    adcc = pd.read_csv('adcc.csv')
    adcc = np.array(adcc)
    # dcc = dcc.iloc[:,1:]
    # dcc = np.array(dcc)
    for i in range(12):
        Data_Set = data[start_date + i*Time_delta: end_date + i*Time_delta]        
        Mean_Train_Data_Set = data[pd.datetime(2016,1,1)+ i*Time_delta:pd.datetime(2018,1,1)+ i*Time_delta]
        Cov_Train_Data_Set = data[pd.datetime(2016,1,1)+ i*Time_delta:pd.datetime(2018,1,1)+ i*Time_delta]
        # arma_mean = ARMA_Predict(Mean_Train_Data_Set, order, symbols, t = 30)
        # mlp_mean = MLP_Predict(Mean_Train_Data_Set, Mean_Train_Data_Set, 25, symbols, t=30)
        # mlp_mean = mlp_mean_all[i]
        # mlp_mean = mlp_mean[1:]
        rnn_mean = RNN_Predict(Mean_Train_Data_Set, Mean_Train_Data_Set, symbols, t = 30)
        rnn_mean_all[i] = rnn_mean
        # rnn_mean = rnn_mean_all[i] 
        # psn_mean = PSN_Predict(Mean_Train_Data_Set, Mean_Train_Data_Set, symbols)
        
        # dcc_cov = COV_Predict_DCC(Mean_Train_Data_Set, symbols)
        # dcc_all[i] = dcc_cov
        # dcc_cov = dcc_all[i]
        # dcc_cov = dcc_cov[1:].reshape((5,5))
        # adcc_cov = COV_Predict_ADCC(Mean_Train_Data_Set, symbols)
        adcc_cov = adcc[i]
        adcc_cov = adcc_cov[1:].reshape((5,5))
        # adcc_all[i] = adcc_cov
        # adcc_cov = dcc[i].reshape((5,5))
        # mlp_mean_all[i] = mlp_mean
        # mlp_mean=mlp_mean_all[i]
        # psn_mean_all[i] = psn_mean
        # rnn_mean = rnn_mean_all[i]
        
        #mean = ARMA
        #cov = DCC
        
        #weights = Generate_Weights(mean,cov)
        weights = Generate_Weights(np.array(rnn_mean),np.array(adcc_cov))
        Port_Rets = np.dot(Data_Set,weights)
        Out_of_sample_Returns = np.append(Out_of_sample_Returns, Port_Rets)
    plt.plot(Out_of_sample_Returns)    
    
    Days = len(Out_of_sample_Returns)
    Out_of_sample_Accu_Returns = np.array([0])
    for i in range(Days-1):
        Out_of_sample_Accu_Returns = np.append(Out_of_sample_Accu_Returns,  np.sum(Out_of_sample_Returns[0:i+1]))    
    plt.plot(Out_of_sample_Accu_Returns)
    ''' transform the data to csv file'''
    # psn_adcc_return = Out_of_sample_Accu_Returns
    # Result2 =pd.DataFrame(psn_adcc_return)
    # Result2 = Result2.to_csv('psn_adcc_return.csv')
    # psn_dcc_return = Out_of_sample_Accu_Returns
    # Result2 =pd.DataFrame(psn_dcc_return)
    # Result2 = Result2.to_csv('psn_dcc_return.csv')
    # mlp_adcc_return = Out_of_sample_Accu_Returns
    # Result2 = pd.DataFrame(mlp_adcc_return)
    # Result2 = Result2.to_csv('mlp_adcc_return')
    
    # mlp_dcc_return = Out_of_sample_Accu_Returns
    # Result2 =pd.DataFrame(mlp_dcc_return)
    # Result2 = Result2.to_csv('mlp_dcc_return.csv')
    
    # rnn_adcc_return = Out_of_sample_Accu_Returns
    # Result2 = pd.DataFrame(rnn_adcc_return)
    # Result2.to_csv('rnn_adcc_return.csv')
    # rnn_dcc_return = Out_of_sample_Accu_Returns
    # Results2 = pd.DataFrame(rnn_dcc_return)
    # Results2.to_csv('rnn_dcc_return.csv')
    
    # arma_dcc_return = Out_of_sample_Accu_Returns
    # Results1 = pd.DataFrame(arma_dcc_return)
    # Results1.to_csv("arma_dcc_return.csv")
    # arma_adcc_return = Out_of_sample_Accu_Returns
    # Results1 = pd.DataFrame(arma_adcc_return)
    # Results1.to_csv("arma_adcc_return.csv")
    
    ''' choose a benchmark S&P 500'''
    # sp_price_2018 = pd.read_csv('^GSPC2018.csv')
    # sp_index = sp_price_2018['Date']
    # sp_index.to_csv('sp_index_2018.csv')
    # sp_price_2018 = sp_price_2018['Adj Close']
    # benchmark = sp_price_2018[0]
    # sp_acc_return_2018 = sp_price_2018/benchmark - 1
    # pd.DataFrame(sp_acc_return_2018).to_csv('SP500')
    
    # sp_price_2019 = pd.read_csv('^GSPC2019.csv')
    # sp_index = sp_price_2019['Date']
    # sp_index.to_csv('sp_index_2019.csv')
    # sp_price_2019 = sp_price_2019['Adj Close']
    # benchmark = sp_price_2019[0]
    # sp_acc_return_2019 = sp_price_2019/benchmark - 1
    # pd.DataFrame(sp_acc_return_2019).to_csv('2019SP500.csv')

    
    
    
    '''
    Copula_1 = Gaussian_Copula(data)
    p1 = Copula_1.pdf((0,0,0,0,0))
    c1 = Copula_1.cdf((0,0,0,0,0))
    Var = Copula_1.portfolio_pdf(weights)
    '''
    # w = cp.Variable(5)
    # gamma = cp.Parameter(nonneg=True)
    # ret = (np.array(mean))*w 
    # risk = cp.quad_form(w, np.array(cov))
    # prob = cp.Problem(cp.Maximize(ret - gamma*risk), 
    #            [cp.sum(w) == 1, 
    #             w >= 0, w<=1])
    
    # gamma.value = 6
    # prob.solve()
    # print(w.value)
    
    
    '''
    x = cp.Variable(5)
    mean = np.array(mean)
    cov = np.array(cov)
    port_ret = mean*x 
    port_var = cp.quad_form(x, cov)
    gamma = 6
    objective1 = cp.Maximize(cp.sum_squares(port_ret - gamma*port_var))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective1, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    print(x.value)
    print(constraints[0].dual_value)                   
    '''              
                    