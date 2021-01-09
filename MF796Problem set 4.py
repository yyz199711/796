#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:13:38 2020

@author: Yuyang Zhang
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

symbols = ['AAPL', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AAP', 'AES', 'AFL',
'AMG', 'A', 'GAS', 'ARE', 'APD', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL',
'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO',
 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BAX', 'BBT', 'BDX', 'BBBY',
 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BMY', 'CHRW', 'CA', 'COG',
 'CLX', 'CME', 'CMS', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX',
 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH',
 'DAL', 'XRAY', 'DVN', 'DO', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW' ,
 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NEE',
 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY',
 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 
 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL',
 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR',
 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 
 'ROST', 'RLD', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SPG', 'SWKS',
 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL',
 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT',
 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJX', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA']

symbols = symbols[0 : 129]

start = datetime(2015,1,1)
df = pdr.get_data_yahoo(symbols,start)['Adj Close']
df = df.dropna(axis = 1, how = 'any')

df1 = np.log(df/df.shift(1))

df1 = df1.dropna(axis=0, how ='any')

df2 = np.matrix(df1).T

cov = np.cov(df2)

w,v = np.linalg.eig(cov)

u,s,vh=np.linalg.svd(cov)


def account(u, s, vh, cov, k) :
    
    for i in range(len(s)) :
        a = np.diag(s)
        index = [j for j in range(i+1, len(s))]
        a[index, index] = 0
        app_var = np.diag(np.matrix(u) * a * np.matrix(vh))
        var = np.diag(cov)
        error = 1 - np.sum(app_var) / np.sum(var)
        if error <= k :
            return i + 1


i_1 = account(u, s, vh, cov, 0.5)       
i_2 = account(u, s, vh, cov, 0.1)

# s_1 = s[0 : i_2]


pca=PCA(n_components=61)
newdf=pca.fit_transform(df1)

inverse_1 = np.linalg.inv(np.matrix(newdf).T * np.matrix(newdf))
residual = np.matrix(df1) - np.matrix(newdf) * inverse_1 * np.matrix(newdf).T * np.matrix(df1)
index = df1.index
plt.plot(index, residual)
plt.xlabel('Time')
plt.ylabel('Residual return')
plt.show()
residual1 = residual.T[6]
plt.plot(index, residual1.T)
plt.xlabel('Time')
plt.ylabel('Residual return')
plt.show()

cov_inverse = np.linalg.pinv(cov,rcond = s[61] / s[0])
G = []
for i in range(234) :
    if i < 134 :
        h = 1
    else :
        h = 0
    G += [h]

G = np.matrix(G).reshape((2, 117))

GC_inverseG = G * cov_inverse * G.T

inverse = np.linalg.inv(GC_inverseG)

e_return = np.matrix(df1.mean(axis=0)).reshape((117,1))
lamda = inverse * (G * cov_inverse * e_return - 2 * 2 * np.matrix([1,0.1]).reshape((2,1))) 
omega = 1 / (2 * 2) * cov_inverse * (e_return - G.T * lamda)

negative = sum([1 for i in omega if i < 0])






































 