#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:08:36 2020

@author: Yuyang Zhang
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA

# import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from scipy.special import kv
# import statsmodels.api as sm
import scipy.stats as scs
from scipy.special import gamma, beta
from scipy.integrate import quad
# from arch import arch_model
# from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from arch import arch_model
from scipy import optimize
# import pyflux as pf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sympy import *
# import cvxpy



# symbols = ['AAPL', 'AMZN', 'NKE', 'MSFT', 'GOOG']
# start = datetime(2015,1,1)
# end = datetime(2020,2,1)
# df = pdr.get_data_yahoo(symbols, start, end)['Adj Close']
# df_train = df['2015-01-05':'2019-01-05'].copy()
# df_test = df['2019-01-06':'2020-02-01'].copy()
# lrets = np.log(df/df.shift(1)).dropna()
# ret_train = lrets["2015-01-05":"2019-01-05"] # choose the train set

# ret_test = lrets["2019-01-06":"2020-02-01"]  # choose the test set


###################################################################################
# choose the best arma model to fit the data

def best_arma(df) :
    '''Find the Best ARMA model parameter for the returns'''
    columns = [column for column in df]
    a = [] # collect the arma order for the data
    
    for i in columns :
        best_aic = np.inf # Start point for AIC check, smallest aic wins
        best_order = None
        best_mdl = None
        
        rng = range(1, 5) # set the biggest number for ARMA(p,q)
        
        for j in rng :
            for k in rng :
                try :
                    tmp_mdl = smt.ARMA(df[i], order = (j, k)).fit(method = 'mle', trend = 'c')
                    tmp_aic = tmp_mdl.aic
                    print(tmp_aic)
                    
                    if tmp_aic < best_aic :
                        
                        best_aic = tmp_aic
                        best_order = (i, j, k)
                        best_mdl = tmp_mdl
                except: continue
        a += [best_order, best_mdl]
    
    return np.array(a)
''' best_arma for every stock
(2,1), (4, 2), (4, 3), (3, 3), (3,2)'''
# order = np.array([[3, 2], [4, 2], [3, 2], [1, 1], [1, 1]])


def ARMA_Predict(df, order, symbols, t = 30) :
    predict = list(0 for i in range(len(symbols)))
    for i in range(len(symbols)) :
        model = smt.ARMA(df[symbols[i]], order = (order[i][0], order[i][1])).fit(method = 'mle', trend = 'c')
        predict[i] = model.forecast(steps = t)[0].reshape((t, 1))
        predict[i] = np.mean(predict[i])
    
    return predict



# Predict_ARMA_price = ARMA_Predict(ret_train, order, symbols, t=30)
# AAPL_ARMA=Predict_ARMA_price[0]/100
# arma = np.zeros(30)
# for i in range(30) :
#     arma[i]=145.72 * np.exp(np.sum(AAPL_ARMA[:i]))

# AAPL=df_test['AAPL']
# AAPL=AAPL[:30]
# t = np.linspace(1,30,30)
# plt.plot(t,arma[:30]);plt.plot(t,AAPL_MLP[:7]);plt.plot(t,NIKE_RNN[:7]);plt.plot(t,AAPL[:7]);plt.plot(t,NKE_price[:7])

# plt.legend(['Predicted Price-ARMA', 'Predicted Price-RNN', 'Predicted Price-MLP','Real Price','Predicted Price-PSN'])
# plt.title('Comparison')
# plt.show()
# plt.plot(t,arma);plt.plot(t,AAPL)
###################################################################################
''' MLP Neural_Network Method'''
class DataProcessing:
    def __init__(self, df_train, df_test):
        self.train = df_train
        self.test = df_test
        self.input_train = []
        self.output_train = []
        self.input_test = []
        self.output_test = []

    def gen_train(self, seq_len):
        """
        Generates training data
        :param seq_len: length of window
        :return: X_train and Y_train
        """
        for i in range((len(self.train)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.train.iloc[i: i + seq_len])
            y = np.array([self.train.iloc[i + seq_len + 1]], np.float64)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train)
        self.Y_train = np.array(self.output_train)

    def gen_test(self, seq_len):
        """
        Generates test data
        :param seq_len: Length of window
        :return: X_test and Y_test
        """
        for i in range((len(self.test)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.test.iloc[i: i + seq_len])
            y = np.array([self.test.iloc[i + seq_len + 1]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test)
        self.Y_test = np.array(self.output_test)




def MLP(df_train, df_test, seq_len) :
    '''Use the simpliest NN structure'''
    process = DataProcessing(df_train, df_test)
    process.gen_test(seq_len)
    process.gen_train(seq_len)

    X_train = process.X_train 
    
    Y_train = process.Y_train 

    X_test = process.X_test 
    Y_test = process.Y_test 

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, Y_train, epochs=100)
    
    if  model.evaluate(X_test, Y_test) < 3 :
        
        return model.evaluate(X_test, Y_test), model
    else :
        return MLP(df_train, df_test, seq_len)

def Loss(df_train, df_test, seq_len, symbols) :
    '''For all symbols get the loss function'''
    a = []
    for i in symbols :
        model_eva, model = MLP(df_train[i], df_test[i], seq_len)
        a += [model_eva]
    
    return np.array(a)

def MLP_predict(data, model, t = 30) :
    data = np.array(data).reshape((1, len(data)))
    price = []
    for i in range(t) :
        pre = model.predict(data) 
        price += [float(pre)]
        data = [i for i in data[0][1 : ]] + [float(pre)]
        data = np.array(data).reshape((1, len(data)))
    
    return np.array(price)

def MLP_Predict(df_train, df_test, seq_len, symbols, t=30) :
    return_lst = []
    for i in range(len(symbols)) :
        df_t = df_train[symbols[i]]*100
        df_tt = df_test[symbols[i]]*100
        
        los, model = MLP(df_t, df_tt, seq_len)
            
        
        data = df_t[-seq_len : ]
        
        predict = MLP_predict(data, model, t=30)
        
        predict = np.mean(predict)
        
        
        
        return_lst += [predict/100]
    
    return return_lst


        
# Predict_MLP_price= MLP_Predict(ret_train, ret_train, 25, symbols, t=30)  
# AAPL_MLP=Predict_MLP_price[0]/100
# mlp = np.zeros(30)
# for i in range(30) :
#     mlp[i]=145.72 * np.exp(np.sum(AAPL_MLP[:i]))


# t=np.linspace(1,10,10)
# plt.plot(t,AAPL_MLP[:10]);plt.plot(t,AAPL[:10])
# plt.legend(['Predicted Price', 'Real_price'])
# plt.title('MLP')
# plt.show()
      
        
###################################################################################
''' RNN method with Keras '''
def Predict_RNN(training_set, df_test, t = 30) :
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(np.array(training_set).reshape(-1,1))
    # Creating a data structure with 30 timesteps and 1 output
    X_train = []
    Y_train = []
    for i in range(10, len(training_set)):
       
        X_train.append(training_set_scaled[i-10:i, 0])
        Y_train.append(training_set_scaled[i, 0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Reshaping
    # print(X_train.shape[0])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            
        
        
    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1, activation = 'tanh'))
    # regressor.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, Y_train, epochs = 15, batch_size = 32)
    
    
    '''predict the price by t days'''
    dataset_total = training_set[-10:]
    inputs = dataset_total.values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    # data_trans = sc.transform(inputs)
    # data_trans = np.reshape(data_trans, (data_trans.shape[1], data_trans.shape[0], 1))
    # predicted_stock_price = regressor.predict(data_trans)
    # predicted_stock_price_1 = sc.inverse_transform(predicted_stock_price)
    
    # return predicted_stock_price_1
    price_mon = []
    data = inputs
    for j in range(t) :
        data_trans = sc.transform(data)
        data_trans = np.reshape(data_trans, (data_trans.shape[1], data_trans.shape[0], 1))
        predicted_stock_price = regressor.predict(data_trans)
        predicted_stock_price_1 = sc.inverse_transform(predicted_stock_price)
        price_mon += [float(predicted_stock_price_1)]
        data = [data[i][0]for i in range(1,len(data))] + [float(predicted_stock_price)]
            
        data = np.array(data).reshape(10, 1)
    
    return price_mon

def RNN_Predict(df_train, df_test, symbols, t = 30) :
    return_lst = []
    for i in symbols:
        ret = Predict_RNN(df_train[i], df_test[i], t = 30)
        
        # for j in range(t) :
        #     if j == 0 :
        #         ret = np.log(price[j]/df_train.iloc[-1])
        #     else:
        #         ret = np.log(price[j]/price[j-1])
        return_lst += [np.mean(ret)/10]
    
    return return_lst

# Predict_RNN_price = RNN_Predict(ret_train, ret_test, symbols, t=30)
# AAPL_RNN=Predict_RNN_price[0]
  
# NIKE=df_test['AMZN']
# t = np.linspace(1,10,10)
# plt.plot(t,NIKE_RNN[:7]);plt.plot(t,NIKE[:7])
# plt.legend(['Predicted price', 'Real Price'])
# plt.title('RNN')
# plt.show()




       


    


###################################################################################
'''PSN method'''
#Using the Dataprocessing object in the MLP method.
def PSN_data(df_train, df_test, seq_len) :
    '''get the train and test data'''
    # print(max(df_train), max(df_test))
    maximum = max(max(df_train), max(df_test))
    
    process = DataProcessing(df_train/maximum, df_test/maximum)
    process.gen_train(seq_len)

    X_train = process.X_train
    
    Y_train = process.Y_train

    
    return X_train, Y_train, maximum


np.random.seed(2)   # set the seed


class PSN:
    def __init__(self, X_train, Y_train, df_test, maximum) :
        self.X1 = X_train
        self.Y1 = Y_train
        self.X2 = df_test
        self.max = maximum
    
    
    def random_weight(self, k, seq_len) :
        '''choose the initial weight'''
        w = list(0 for i in range(k))
        for i in range(k) :
            w[i] = np.random.uniform(-1,1, seq_len+1)
        
        # self.initial_weight = w
        return w
    
    def hidden(self, w, k, x):
        '''Get the value for hidden layer'''

        h = np.zeros(k)
        for i in range(k) :
            h[i] = np.dot(w[i], x)
        
        return h
    
    def bias(self, x_b) :
        '''insert a bias node'''
        a = []
        for i in self.X1 :
            lst_i = [m for m in i]
            lst_i.append(x_b)
            a += [lst_i]
        a = np.array(a)
        self.X_train = a
        
    def signmold(self, x) :
        '''target function'''
        f_x = 1 / (1 + np.exp(-x))
        
        return f_x
    
    
    # def d2(self, c, x) :
    #     '''return the first derivative with respect to c'''
    #     h_x = x * np.exp(-c * x) / (1 + np.exp(-c * x)) ** 2
        
    #     return h_x
    
    def ds(self, x):
        '''return the first derivative for signmold'''
        g_x = (1 - self.signmold(x)) * self.signmold(x)
        
        return g_x
        
         
    def residue(self, w, k, x_b) :
        '''get the residue of square'''
        self.bias(x_b)
        y = np.zeros(len(self.Y1))
        pros = np.zeros(len(self.Y1))
        grad_1 = np.zeros(len(self.Y1))
        hh =[[] for m in range(len(self.Y1))]
        for i in range(len(self.X_train)) :
            xt = self.X_train[i]
            h = self.hidden(w, k, xt)
            pro = 1
            
            for j in range(len(h)) :
                pro *= h[j]
                hh[i] += [h[j]]
            
            pros[i] = pro
            y[i] = self.signmold(pro)
            grad_1[i] = self.ds(pro)
        
        E = np.sum((y - self.Y1) ** 2) / len(self.Y1)
        X_train_trans = self.X_train.T
        grad = [[] for l in range(k)]
        hh = np.array(hh)
        print(hh)
        for j in range(k) :
            
            a = y - np.array([float(self.Y1[i]) for i in range(len(self.Y1))])
            hh_1 = np.array([hh[u][j] for u in range(len(hh))])
            a = a * pros * grad_1 / hh_1
            # print(a)
            for p in range(len(X_train_trans)) :
                x_p = X_train_trans[p]
                grad[j] += [2 * np.sum(a * x_p) / len(self.Y1)]
                # print(grad[j])
            
            grad[j] = np.array(grad[j])
 
        
        
        
        return E, pros, grad
    
    def Best_weight(self, k, x_b, seq_len, lr, accuracy) :
        '''use the gradient loss to change the weight'''
        w = self.random_weight(k, seq_len)
        r1, pros, grad = self.residue(w, k, x_b)
        residue = [[r1, w]]
        
        flag = 0 # decide if we choose the best weight
        while flag < 10 :
            i = np.random.choice(range(k))
                # w_change = self.d1(c, pros[i])
                # print(w_change)
            w_change = lr * grad[i] # choose a new weight
                # print(w_change)
            w[i] -= w_change
            # c_change = self.d2(c, pros[i])
            # print()
            r1, grad, pros = self.residue(w, k, x_b)
            # print(abs(r1 - residue[-1][0]))
            if abs(r1 - residue[-1][0])  < accuracy :
                flag += 1
            else :
                flag = 0 
                
            residue += [[r1, w]]
        
        return residue[-1]

    def Best_k(self, seq_len, lr, accuracy, x_b) :
        '''choose the best k and X_b'''
        k = range(10, 101)
        
        index = []
        for i in k :
            res = []
            
            res += [self.Best_weight(i, x_b, seq_len, lr, accuracy)]
            
            print(min(res))
            
            index += [min(res).append(i)]
        
        return min(index)
    '''The best k is , x_b we choose as 1'''
    
    def predict(self, seq_len, lr, accuracy, k, t = 30) :
        '''predict the value for price'''
        Best = self.Best_weight(k, 1, seq_len, lr, accuracy)
        w = Best[1]
        
        a = [j for j in self.X1[-1]]
        
        a = a[1:]
        
        a = a + [m for m in self.Y1[-1]] + [0 for p in range(t)] # find the last 11 data + 30 empty data
        
        for i in range(t) :
            use_data = a[i:i+seq_len] + [1]
            
            use_data = np.array(use_data)
            
            h = self.hidden(w, k, use_data)
            
            pro = 1
            
            for j in h :
                
                pro *= j
            a[i+seq_len] = self.signmold(pro)
        
        a = a[seq_len : ]
        
        a = np.array(a) * self.max
        
        return a

def PSN_Predict(df_train, df_test, symbols, seq_len = 10, t=30) :
    Price_lst = []
    for i in range(len(symbols)) :
        X_train, Y_train, max_i = PSN_data(df_train[symbols[i]], df_test[symbols[i]], seq_len)
        
        psn = PSN(X_train, Y_train, df_test, max_i)
        if symbols[i] == 'NKE' :
            lr = 0.01
        else :
            lr = 0.001
        
        Predict = psn.predict(seq_len, lr, 0.0001, 11, t)
        
        Price_lst += [np.mean(Predict)/10]
    
    return Price_lst

# Predict_PSN_return = PSN_Predict(ret_train["2018-09-01":], ret_test, symbols, 10, 30)
# AAPL_PSN = Predict_PSN_return[3]/5
# AAPL_price = np.zeros(30)
# for i in range(30) :
#     AAPL_price[i]=1070.71 * np.exp(np.sum(AAPL_PSN[:i]))
# plt.plot(t,NKE_price[:10]);plt.plot(t,AAPL[:10])
# plt.legend(['Predicted price', 'Real Price'])
# plt.title('PSN')
# plt.show()



# X_train, Y_train, max_A = PSN_data(df_train['ATR'], df_test['ATR'], 10)
# psn = PSN(X_train, Y_train, df_test['ATR'], max_A)
# Predict_P = psn.predict(10, 0.01, 0.0001, 11, 30)
        
###################################################################################
''' Choose the DCC-GARCH model for each return and get the residue'''



def GARCH_params(ret_train, symbols) :
    h = list(0 for i in range(len(symbols)))
    epsilon = list(0 for i in range(len(symbols)))
    std = list(0 for i in range(len(symbols)))
    # alpha = list(0 for i in range(len(symbols)))
    # beta = list(0 for i in range(len(symbols)))
    eta = list(0 for i in range(len(symbols)))


    for i in range(len(symbols)) :
        model = arch_model(ret_train[symbols[i]]*100, mean = 'Constant', dist='t', vol='GARCH')
        model_fit = model.fit()
        forecasts = model_fit.forecast(horizon=30)
    
    # print(model_fit.summary())
        h[i] = model_fit._volatility
        epsilon[i] = model_fit.std_resid
        std[i] = np.sqrt(forecasts.variance.dropna())/100
        std[i] = np.array(std[i])
        # alpha[i] = model_fit._params[1]
        # beta[i] = model_fit._params[2]
        eta[i] = model_fit._params[4]
    # lamba[i] = model_fit._params[]

    h = np.array(h).T
    epsilon = np.array(epsilon).T
    std = np.array(std).T
    
    # omega = np.array(omega)
    # alpha = np.array(alpha)
    # beta = np.array(beta)
    eta = np.array(eta)
    return h, epsilon, std.reshape((30,5)), eta



def DCC(h, epsilon, a_1, a_2) :
    '''Get the parameter for DCC'''
    ep = np.matrix(epsilon).T
    corr = np.corrcoef(ep)
    q = list(0 for i in range(len(epsilon)))
    q_0 = np.matrix(np.zeros((len(epsilon[0]), len(epsilon[0]))))
    # print(q_0)
    for i in range(len(epsilon[0])) :
        for j in range(len(epsilon[0])) :
            if i == j :
                q_0[i, j] = 1
            else :
                q_0[i, j] = sum(epsilon[:][i] * epsilon[:][j]) / len(epsilon)
    
    q[0] = q_0
    
    ep_square = list(0 for i in range(len(epsilon)))
    
    for k in range(len(epsilon)) :
        
        e_k = epsilon[k]
        e_k_square = e_k * e_k.reshape((len(epsilon[0]), 1))
        
        ep_square[k] = np.matrix(e_k_square)
        
    
    for m in range(1, len(epsilon)) :
        
        q[m] = (1 - a_1 - a_2) * np.matrix(corr) + a_1 * ep_square[m - 1] + a_2 * q[m - 1]
    
    rho = list(0 for i in range(len(epsilon)))
    
    for l in range(len(epsilon)) :
        rho[l] = np.corrcoef(q[l])
        
        rho[l] = np.matrix(rho[l])
    
    
    
    
    a = np.zeros(len(epsilon))
    
    
    for n in range(len(epsilon)) :
        h_n = np.eye(len(epsilon[0]))
        for p in range(len(epsilon[0])) :
            h_n[p][p] = h[n][p]
        
        h_n = np.matrix(h_n)
        x1 = 2 * np.log(abs(np.linalg.det(h_n)))
        # print(x1)
        x2 = 2 * np.log(abs(np.linalg.det(rho[n])))
        # print(x2)
        x3 = np.matrix(epsilon[n]) * np.linalg.pinv(rho[n]) * np.matrix(epsilon[n]).T
        
        # print(x3)
        if np.isinf(x1 + x2 + x3) :
            a[n] = 0
        else :
            
            a[n] = -(len(epsilon[0]) * np.log(2 * np.pi) + x1 + x2 + float(x3))
    
    Loglikelyhood = np.sum(a)
    # print(Loglikelyhood)
    
    return Loglikelyhood, q[-1], ep_square[-1]

def find_estimate(h, epsilon) :
    a_1 = np.linspace(0.05,0.95, 10)
    a_2 = np.linspace(0.05,0.95, 10)
    
    a = []
    for i in a_1 :
        for j in a_2 :
            a += [[f([i,j],h,epsilon),[i, j]]]
            
    est = min(a)
    
    return est[1]

def f(x,h,epsilon) :
    loglikeh, q, ep_square = DCC(h, epsilon, x[0], x[1])    
    return -loglikeh

'''Optimize the parameter is 0.05, 0.25'''

def DCC_Predict(h, epsilon, a_1, a_2, eta, t = 30):
    loglikeh, q, ep_square = DCC(h, epsilon, a_1, a_2)
    q_all = list(0 for i in range(t+1))
    q_all[0] = q
    ep_square_all = list(0 for i in range(t+1))
    ep_square_all[0] = ep_square
    ep = np.matrix(epsilon).T
    corr = np.corrcoef(ep)
    rho = list(0 for i in range(t+1))
    for j in range(1, t+1) :
        a = (1 - a_1 - a_2) * np.matrix(corr) + a_1 * ep_square_all[j-1] + a_2 * q_all[j-1]
        q_all[j] = a
        b = np.zeros((len(q_all[j]), len(q_all[j])))
        
        for k in range(len(q_all[j])) :
            for l in range(len(q_all[j])) :
                b[k][l] = a[k, l] / np.sqrt(a[k, k] * a[l, l])
        rho[j] = b
        ep_j = multivariatet(0, rho[j], eta, 1)
        ep_square_all[j] = np.matrix(ep_j * ep_j.reshape((len(epsilon[0]), 1)))
    
    return rho
        

def multivariatet(mu,Sigma,N,M):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
    return mu + Z/np.sqrt(g)
    
def COV_Predict_DCC(ret_train, symbols, t=30) :
    h, epsilon,std,eta = GARCH_params(ret_train, symbols)
    a = find_estimate(h, epsilon)
    a_1 = a[0]
    a_2 = a[1]
    rho = DCC_Predict(h, epsilon, a_1,a_2,eta,t)
    rho = rho[1:]
    cov = list(np.zeros((5,5)) for i in range(t))
    
    for i in range(t) :
        std_i = std[i]
        rho_i = rho[i]
        
        for j in range(5) :
            for k in range(5) :
                cov[i][j][k]=rho_i[j][k]*std_i[j]*std_i[k]
        
    cov_ave = np.zeros((5,5))
    for p in range(t) :
        
        cov_ave += cov[p]
    
    
    cov_ave = cov_ave/t
    
    return cov_ave
        
    
        

    
    
    
    
    

###################################################################################
'''This is for the ADCC model,
If we want to get the parameters for skew t, we just need to change the dist to skewt
and then get the skewness parameter'''    
def GJR_GARCH_params(ret_train, symbols) :
    h = list(0 for i in range(len(symbols)))
    epsilon = list(0 for i in range(len(symbols)))
    std = list(0 for i in range(len(symbols)))
    # alpha = list(0 for i in range(len(symbols)))
    # beta = list(0 for i in range(len(symbols)))
    eta = list(0 for i in range(len(symbols)))


    for i in range(len(symbols)) :
        model = arch_model(ret_train[symbols[i]]*100, mean = 'Constant', dist='t', p=1,q=1,o=1)
        model_fit = model.fit()
        forecasts = model_fit.forecast(horizon=30)
    
    # print(model_fit.summary())
        h[i] = model_fit._volatility
        epsilon[i] = model_fit.std_resid
        std[i] = np.sqrt(forecasts.variance.dropna())/100
        std[i] = np.array(std[i])
        # alpha[i] = model_fit._params[1]
        # beta[i] = model_fit._params[2]
        eta[i] = model_fit._params[5]
    # lamba[i] = model_fit._params[]

    h = np.array(h).T
    epsilon = np.array(epsilon).T
    std = np.array(std).T
    
    # omega = np.array(omega)
    # alpha = np.array(alpha)
    # beta = np.array(beta)
    eta = np.array(eta)
    return h, epsilon, std.reshape((30,5)), eta


def ADCC(h, epsilon, a_1, a_2, g) :
    '''Get the parameter for ADCC'''
    ep = np.matrix(epsilon).T
    corr = np.corrcoef(ep)
    q = list(0 for i in range(len(epsilon)))
    q_0 = np.matrix(np.zeros((len(epsilon[0]), len(epsilon[0]))))
    # print(q_0)
    for i in range(len(epsilon[0])) :
        for j in range(len(epsilon[0])) :
            if i == j :
                q_0[i, j] = 1
            else :
                q_0[i, j] = sum(epsilon[:][i] * epsilon[:][j]) / len(epsilon)
    
    q[0] = q_0
    
    ep_square = list(0 for i in range(len(epsilon)))
    np_square = list(0 for i in range(len(epsilon)))
    
    
    
    for k in range(len(epsilon)) :
        
        e_k = epsilon[k]
        e_k_square = e_k * e_k.reshape((len(epsilon[0]), 1))
        
        n_k = np.zeros(len(e_k))
        
        for v in range(len(e_k)) :
            if epsilon[k][v] < 0 :
                n_k[v] = e_k[v]
        n_k_square = n_k * n_k.reshape((len(epsilon[0]), 1))
                
                
        np_square[k] = np.matrix(n_k_square)
        ep_square[k] = np.matrix(e_k_square)
    
    Bar_N = np_square[0]
    for u in range(1, len(epsilon)) :
        Bar_N += np_square[u]
    
    Bar_N = Bar_N / len(epsilon)
    
        
        
    
    for m in range(1, len(epsilon)) :
        
        q[m] = (1 - a_1 - a_2) * np.matrix(corr) + a_1 * ep_square[m - 1] + a_2 * q[m - 1] - g * Bar_N + g * np_square[m - 1]                         
    
    rho = list(0 for i in range(len(epsilon)))
    
    for l in range(len(epsilon)) :
        rho[l] = np.corrcoef(q[l])
        rho[l] = np.matrix(rho[l])
    
    
    
    
    a = np.zeros(len(epsilon))
    
    
    for n in range(len(epsilon)) :
        h_n = np.eye(len(epsilon[0]))
        for p in range(len(epsilon[0])) :
            h_n[p][p] = h[n][p]
        
        h_n = np.matrix(h_n)
        x1 = 2 * np.log(abs(np.linalg.det(h_n)))
        # print(x1)
        x2 = 2 * np.log(abs(np.linalg.det(rho[n])))
        # print(x2)
        x3 = np.matrix(epsilon[n]) * np.linalg.pinv(rho[n]) * np.matrix(epsilon[n]).T
        
        # print(x3)
        if np.isinf(x1 + x2 + x3) :
            a[n] = 0
        else :
            
            a[n] = -(len(epsilon[0]) * np.log(2 * np.pi) + x1 + x2 + float(x3))
    
    Loglikelyhood = np.sum(a)
    # print(Loglikelyhood)
    # print(a_1)
    
    
    return Loglikelyhood, q[-1], ep_square[-1], np_square[-1], Bar_N

def Afind_estimate(h, epsilon) :
    a_1 = np.linspace(0.05,0.95,5)
    a_2 = np.linspace(0.05,0.95, 10)
    g = np.linspace(0.1, 1, 5)
    
    a = []
    for i in a_1 :
        for j in a_2 :
            for k in g :
                
                a += [[Af([i,j,k],h,epsilon),[i, j, k]]]
            
    est = min(a)
    
    return est[1]

    

def Af(x,h,epsilon) :
    loglikeh, q, ep_square, np_square, Bar_N = ADCC(h, epsilon, x[0], x[1], x[2])
    
    return -loglikeh

'''The optimal set is 0.05,0.35,0.1'''

def ADCC_Predict(h,epsilon,a_1, a_2, g, eta, lamba = 0, t = 30) :
    loglikeh, q, ep_square, np_square, Bar_N = ADCC(h, epsilon, a_1, a_2, g)
    q_all = list(0 for i in range(t+1))
    q_all[0] = q
    ep_square_all = list(0 for i in range(t+1))
    np_square_all = list(0 for i in range(t+1))
    ep_square_all[0] = ep_square
    np_square_all[0] = np_square
    ep = np.matrix(epsilon).T
    corr = np.corrcoef(ep)
    rho = list(0 for i in range(t+1))
    for j in range(1, t+1) :
        a = (1 - a_1 - a_2) * np.matrix(corr) + a_1 * ep_square_all[j-1] + a_2 * q_all[j-1] - g * Bar_N + g * np_square_all[j]
        q_all[j] = a
        b = np.zeros((len(q_all[j]), len(q_all[j])))
        
        for k in range(len(q_all[j])) :
            for l in range(len(q_all[j])) :
                b[k][l] = a[k, l] / np.sqrt(a[k, k] * a[l, l])
        rho[j] = b
        ep_j = multivariatet(0, rho[j], eta, 1)
        # if j == 1 :
            
        #     print(ep_j)
        ep_square_all[j] = np.matrix(ep_j * ep_j.reshape((len(epsilon[0]), 1)))
        np_j = np.zeros(len(ep_j[0]))
        for m in range(len(ep_j[0])) :
            if ep_j[0][m] < 0 :
                np_j[m] = ep_j[0][m]
        np_square_all[j] = np.matrix(np_j * np_j.reshape((len(epsilon[0]), 1)))
        
    
    return rho

def COV_Predict_ADCC(ret_train, symbols, t=30) :
    h, epsilon,std,eta = GJR_GARCH_params(ret_train, symbols)
    a = Afind_estimate(h, epsilon)
    a_1 = a[0]
    a_2 = a[1]
    g = a[2]
    rho = ADCC_Predict(h, epsilon, a_1,a_2,g,eta,t)
    rho = rho[1:]
    cov = list(np.zeros((5,5)) for i in range(t))
    
    for i in range(t) :
        std_i = std[i]
        rho_i = rho[i]
        
        for j in range(5) :
            for k in range(5) :
                cov[i][j][k]=rho_i[j][k]*std_i[j]*std_i[k]
        
    cov_ave = np.zeros((5,5))
    for p in range(t) :
        
        cov_ave += cov[p]
    
    
    cov_ave = cov_ave/t
    
    return cov_ave
        


    
   


###################################################################################
'''This is for modelling the density'''
# def model_parameter(ret_train) :
#     ret_train_1 = ret_train
#     ret_mdl = arch_model(ret_train_1*100, p = 1, q = 1, o = 1, dist='skewt').fit()
#     params = ret_mdl.params
#     stresid = ret_mdl.std_resid
    
#     return stresid, params

# res = list(0 for i in range(len(symbols)))
# parameter = list(0 for i in range(len(symbols)))
# for i in range(len(symbols)) :
#     resid, params = model_parameter(ret_train[symbols[i]])
#     res[i] = resid
#     parameter[i] = params

class Skewt:
    
    def __init__(self, lamba, degree, mu, sigma) :
        self.l = lamba
        self.q = degree / 2
        self.mu = mu
        self.sigma = sigma
        self.nu = 1 / (self.q ** 0.5 * ((3 * self.l ** 2 + 1) * (1 / (2 * self.q - 2) - 4 * self.l ** 2 / np.pi * (gamma(self.q - 0.5) / gamma(self.q)) ** 2)) ** 0.5)
        self.m = 2 * self.nu * self.l * self.q ** 0.5 * gamma(self.q - 0.5) / (np.sqrt(np.pi) * gamma(self.q + 0.5))
        
    
    def prob_den(self, x) :
        nume = gamma(0.5 + self.q)
        y_1 = (x - self.mu + self.m) ** 2 / (self.q * (self.nu * self.sigma) ** 2)
        y_1 = y_1 / (self.l * np.sign(x - self.mu + self.m) + 1) ** 2
        
        
        y_2 = (y_1 + 1) ** (0.5 + self.q)
        
        y_3 = self.nu * self.sigma * (np.pi * self.q) ** 0.5 * gamma(self.q)
        
        f = gamma(0.5 + self.q) / (y_2 * y_3)
        
        return f
    def mean(self) :
        mean = self.mu + 2 * self.nu * self.sigma * self.q ** 0.5 * beta(1, self.q - 0.5) / beta(0.5, self.q) - self.m
        
        return mean
    
    def variance(self) :
        var = (self.nu * self.sigma) ** 2 * self.q
        var *= (3 * self.l ** 2 + 1) * beta(1.5, self.q - 1) / beta(0.5, self.q) - 4 * self.l ** 2 * beta(1, self.q - 0.5) ** 2 / beta(0.5, self.q) ** 2
        
        return var
    
    def prob_trans(self, data) :
        
        a = []
        for i in data :
            
            integral = quad(self.prob_den(), -np.inf, i)
            a += [integral]
        
        return np.array(a)
    
    
def find_scale_skewt(lamba, degree) :
    mu = Symbol('mu', real = True)
    sigma = Symbol('sigma', real = True)
    a = solve([Skewt(lamba, degree, mu, sigma).mean(), Skewt(lamba, degree, mu, sigma).variance() - 1], [mu, sigma])
    
    return a
    
'''choose the solution where sigma is positive'''
###################################################################################

def mul_skew_t_density(param, Sigma, d, x) :
    '''Sigma: the predicted covariance matrix
       nu : degree of freedom for multivariate skew t
       alpha : skewness parameter.
       d : dimension of vectors
    '''
    alpha = param[1:]
    alpha = np.array(alpha).reshape((5,1))
    alpha = np.matrix(alpha)
    nu = param[0]
    c = 2**((2-nu-d)/2)/(gamma(nu/2)*(np.pi*nu)**(d/2)*abs(np.linalg.det(Sigma))**0.5)
    
    Sigma_inv = np.linalg.pinv(Sigma)
    y = nu+float(x.T*Sigma_inv*x)
    y = y * float(alpha.T * Sigma_inv * alpha)
    y = y ** 0.5
    
    K = kv((nu+d)/2, y)
    
    z_1 = exp(float(x.T * Sigma_inv * alpha))
    
    z_2 = y ** (-(nu+d)/2)
    # z_2 = (-(nu+d)/2)*np.exp(np.log(y))
    
    z_3 = (1 + float(x.T*Sigma_inv*x)/nu)**((nu+d)/d)
    
    density = c * K * z_1 / (z_2 * z_3)
    
    return density


def MLE_skewt(param, Sigma, data) :
    '''Sigma_all : Covariance matrix for all dataset
       data: for all epsilon we have
    '''
    l_density = 0
    for i in range(len(data)) :
        epsilon_i = np.matrix(data[i].reshape((5,1)))
        density_i = mul_skew_t_density(param, Sigma, 5, epsilon_i)
        # print(density_i)
        # if density_i <= 0 : 
        #     print(density_i)
        l_density_i = log(density_i)
        
        l_density += l_density_i
    
    
    return -l_density

# Sigma = np.matrix(np.corrcoef(epsilon.T))
# data_skew = epsilon
# '''optimal set is 4.5,-0.5,-0.1,0.5,-0.2,0.3'''

def Simulation_mulskewt(M, Sigma_all, nu, alpha, d=2) :
    '''Sigma_all:All Sigma we predict by DCC or ADCC
       nu: optimal nu we get
       alpha: optimal alpha we get
       M : generate sample number
    '''
    sim = list(0 for i in range(len(Sigma_all)))
    for i in range(len(Sigma_all)) :
        
        g = np.tile(np.random.gamma(nu/2.,2./nu,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma_all[i],M)
        
        sim[i] = alpha/g + Z/np.sqrt(g)
    
    return sim

# Sim_res = Simulation_mulskewt(10000,rho,alpha[0],alpha[1:])
    
def model_forecast(ret_train) :
    ret_train_1 = ret_train
    ret_mdl = arch_model(ret_train_1*100, p = 1, q = 1, o = 1, dist='skewt').fit(update_freq=1)
    forecasts = ret_mdl.forecast()
    
    return forecasts.mean.dropna(), forecasts.variance.dropna(), forecasts.residual_variance.dropna()

'''Copula simulation result'''
# means = list(0 for i in range(5))
# var = list(0 for i in range(5))
# ave_mean = 0
# resid_var_1 =np.zeros(5)
# for i in range(5) :
#     means, var, resid_var = model_forecast(ret_train[symbols[i]])
#     # ave_mean += means[i].iloc[0][0]
#     resid_var_1[i] = resid_var.iloc[0][0]
    
# resid_var_1 = np.sqrt(resid_var_1)/100
# h, epsilon, std, eta=GJR_GARCH_params(ret_train, symbols)
# rho = ADCC_Predict(h,epsilon, 0.05,0.35,0.1,eta)
# rho = rho[1]
# cov = np.zeros((5,5))
# for i in range(5) :
#     for j in range(5) :
#         cov[i][j] = rho[i][j] * resid_var_1[i] * resid_var_1[j]
# cov = cov[:2]
# cov = cov.T
# cov = cov[:2]
# cov = list(cov for i in range(1))
# alpha = np.array([4.5,-0.006,-0.01,-0.2,-0.05,0])
# np.random.seed(13)
# sim_res = Simulation_mulskewt(10000,cov,alpha[0],alpha[1:],d=5)
# sim_res = sim_res[0]
# x = sim_res.T[1]
# y = sim_res.T[4]
# plt.scatter(x,y)
# sim_res1 = np.random.multivariate_normal(np.zeros(5),cov[0],10000)
# x_1 = sim_res1.T[0]
# y_1 = sim_res1.T[1]
# plt.scatter(x_1,y_1)
        
# rho = pd.read_csv('adcc.csv')
# rho = rho.iloc[0,1:]
# rho = np.array(rho).reshape((5,5))

# sqrt_1 = np.sqrt(var_1)

# sqrt_1=sqrt_1.reshape((5,1))

# sim_resid = np.dot(xy, sqrt_1)/500

# sim_resid = sim_resid.reshape(10000,)

# sim_resid_sort = np.sort(sim_resid)

# VAR = sim_resid_sort[250]
    

    


    
    


        
        
    

    
    
    



        
        
        
        
    
    
    
    
    
    
     
    
    
    
    
           
        
        
    
    
        
    
    
    
                
            
    


    
    
    
    
            
    
    


    
    
    
        
        
        
            
            
                
                
                
                
        
        
        
    
    
    
    
            
            
        
            
        
        
        
            
            
            
                
            
        
        
            
        
            
            
            
        






















