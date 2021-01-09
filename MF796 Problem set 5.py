#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:51:12 2020

@author: Yuyang Zhang
"""
import numpy as np
from scipy.stats import norm
# import scipy.stats as si
from scipy.interpolate import splev, splrep, interp2d, interp1d

import matplotlib.pyplot as plt
import pandas as pd
from a4task796 import *
from a3task796 import *
from scipy.optimize import minimize

class Eurocall:
    
    def __init__(self, s, k, t, sigma, r) :
        self.s = s
        self.k = k
        self.t = t
        self.sigma = sigma
        self.r = r
        self.u = (self.r - self.sigma ** 2 / 2) * self.t + np.log(self.s)
    
    def d_1(self) :
        
        d_1 = (np.log(self.s / self.k) + (self.r + np.power(self.sigma, 2) / 2) * self.t) / (self.sigma * np.sqrt(self.t))
        return d_1
    
    def d_2(self) :
        
        d_2 = self.d_1() - self.sigma * np.sqrt(self.t)
        
        return d_2
    def BScall(self):
    
        call = self.s * norm.cdf(self.d_1()) - self.k * np.exp(-self.r * self.t) * norm.cdf(self.d_2())
    
        return call

def Kcall(sigma, t, delta):
    
    d_1 = norm.ppf(delta)
    
    K = np.exp(np.log(100) + sigma ** 2 * t / 2 - d_1 * sigma * np.sqrt(t))
    
    return K
    

def table() :
    t = [1/12, 0.25]
    delta = [0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1]
    v = [0.3225, 0.2836, 0.2473, 0.2178, 0.2021, 0.1818, 0.1824, 0.1645, 0.1574, 0.1462, 0.1370, 0.1256, 0.1148, 0.1094]
    v = np.array(v).reshape((7, 2))
    
    b = []
    for i in range(len(delta)) :
        
        a = []
        for j in range(len(t)) :
            
            k = Kcall(v[i][j], t[j], delta[i])
            
            a += [k]
        
        b += [a]
    
    b = np.array(b)
    return b
    
def vol_func(k, v) :
    
    # tck = interp1d(k, v, kind='cubic', bounds_error=False, fill_value=(v[0], v[-1]))
    # k_1 = np.arange(0.5, 200, 0.5)
    
    # sigma_1 = tck(k_1)
    
    # plt.plot(k, v, 'o', k_1, sigma_1)
    # plt.xlabel('Strike K')
    # plt.ylabel('Volatility Sigma')
    # plt.show()
    p = np.polyfit(k, v, 1)
    return p
    


def risk_den(f, t) :
    k = np.arange(65, 130, 0.05)
    # p_1, p_2 = vol_func(k, v)
    sigma = f(k)
    for i in range(len(sigma)) :
        if sigma[i] <= 0 :
            sigma[i] = 0.000001
            
    den = []
    for i in range(1, len(k) - 1) :
        c_1 = Eurocall(100, k[i-1], t, sigma[i-1], 0).BScall()
        c_2 = Eurocall(100, k[i], t, sigma[i], 0).BScall()
        c_3 = Eurocall(100, k[i+1], t, sigma[i+1], 0).BScall()
        
        a = (c_1 + c_3 - 2 * c_2)/0.05**2 
        
        den += [a]
    
    kk = k[1 : len(k) - 1]
    
    den = np.array(den)
    
    plt.plot(kk, den)
    plt.xlabel('Strike K')
    plt.ylabel('Density')
    plt.show()
    return (kk, den)
        
        
def risk_den1(sigma, t) :
    k = np.arange(1, 200, 0.5)
    den = []
    for i in range(1, len(k) - 1) :
        c_1 = Eurocall(100, k[i-1], t, sigma, 0).BScall()
        c_2 = Eurocall(100, k[i], t, sigma, 0).BScall()
        c_3 = Eurocall(100, k[i+1], t, sigma, 0).BScall()
        
        a = (c_1 + c_3 - 2 * c_2) / 0.5 ** 2
        
        den += [a]
    
    kk = k[1 : len(k) - 1]
    
    den = np.array(den)
    
    plt.plot(kk, den)
    plt.xlabel('Strike K')
    plt.ylabel('Density')
    plt.show()
    return (kk, den)
       
        
def price(f, l, u, t) :
    n = np.linspace(l, u, 1001)
    ds = (u - l) / 1000
    kk, den = risk_den(f, t)
    f = interp1d(kk, den, 'cubic')
    integral = 0
    for i in range(len(n) - 1) :
        x = n[i] + ds / 2
        phi = f(x)
        # print(phi)
        integral += phi * ds
        
    return integral
        
        
def price1(f, l, u, t) :
    n = np.linspace(l, u, 1001)
    ds = (u - l) / 1000
    kk, den = risk_den(f, t)
    f = interp1d(kk, den, 'cubic')
    integral = 0
    for i in range(len(n) - 1) :
        x = n[i] + ds / 2
        phi = f(x) * (x - l)
        # print(phi)
        integral += phi * ds
        
    return integral        
    
    
def mini(x) :
    global parameter
    k_1 = np.array(parameter[0:9, 1])
    t_1 = parameter[0][0]
    k_2 = np.array(parameter[9:25, 1])
    t_2 = parameter[9][0]
    k_3 = np.array(parameter[25:44, 1])
    t_3 = parameter[24][0]
    t = np.array([t_1,t_2,t_3])
    k = np.array([k_1,k_2,k_3])
    c_1 = np.array(parameter[0:9, 2])
    c_2 = np.array(parameter[9:25, 2])
    c_3 = np.array(parameter[25:44, 2])
    c = np.array([c_1, c_2, c_3])
    
    
    a = 0
    for i in range(3) :
        ff = FTT(x[0], x[1], x[2], x[3], x[4], 267.15, -0.0027, t[i])
        p,run = ff.Heston_fft(1.5, 8, 267.15*2.7, k[i])
        # print(p)
        res = (p - c[i]) ** 2
        res = np.sum(res)
        a += res
        
    return a

def dmini(x) :
    global parameter
    k_1 = np.array(parameter[0:9, 1])
    t_1 = parameter[0][0]
    k_2 = np.array(parameter[9:25, 1])
    t_2 = parameter[9][0]
    k_3 = np.array(parameter[25:44, 1])
    t_3 = parameter[24][0]
    t = np.array([t_1,t_2,t_3])
    k = np.array([k_1,k_2,k_3])
    c_1 = np.array(parameter[0:9, 2])
    c_2 = np.array(parameter[9:25, 2])
    c_3 = np.array(parameter[25:44, 2])
    c = np.array([c_1, c_2, c_3])
    w_1 = np.array(parameter[0:9, 3])
    w_2 = np.array(parameter[9:25, 3])
    w_3 = np.array(parameter[25:44, 3])
    w = np.array([w_1, w_2, w_3])
    
    
    a = 0
    for i in range(3) :
        ff = FTT(x[0], x[1], x[2], x[3], x[4], 267.15, -0.0027, t[i])
        p,run = ff.Heston_fft(1.5, 8, 267.15*2.7, k[i])
        # print(p)
        res = w[i] * (p - c[i]) ** 2
        res = np.sum(res)
        a += res
        
    return a

def delta_heston() :
    h = 0.5
    f1 = FTT(0.12, 0.017, 0.008, 0.9, 0.31, 267.15-h, -0.0027, 0.25)
    p1, run1 = f1.Heston_fft(1.5, 8, 267.15*2.7, 275)
    f2 = FTT(0.12, 0.017, 0.008, 0.9, 0.31, 267.15+h, -0.0027, 0.25)
    p2, run2 = f2.Heston_fft(1.5, 8, 267.15*2.7, 275)
    delta = (p2 - p1) / (2 * h)
    
    
    
    return delta
    
def delta_BS() :
    upper_bound = 1
    lower_bound = 0.0001
    guess = 0.5
    a = Eurocall(267.15, 275, 0.25, guess, -0.0027).BScall()
    b, run = FTT(0.12, 0.017, 0.008, 0.9, 0.31, 267.15, -0.0027, 0.25).Heston_fft(1.5, 8, 267.15*2.7, 275)
    # print(b)
    while abs(a - b) > 0.0001 :
        if a > b :
            upper_bound = guess
            guess = (upper_bound + lower_bound) / 2
            
        if a < b :
            lower_bound = guess
            guess = (lower_bound + upper_bound) / 2
                
            
            
        a = Eurocall(267.15, 275, 0.25, guess, -0.0027).BScall()
            # print(guess, a)
        
    d1 = Eurocall(267.15, 275, 0.25, guess, -0.0027).d_1()
    # print(Eurocall(267.15, 275, 0.25, guess, -0.0027).BScall())
    
    return norm.cdf(d1), guess
    
def vega_heston() :
    h = 0.001
    f1 = FTT(0.12, 0.017-h, 0.008, 0.9, 0.31-h, 267.15, -0.0027, 0.25)
    p1, run1 = f1.Heston_fft(1.5, 8, 267.15*2.7, 275)
    f2 = FTT(0.12, 0.017+h, 0.008, 0.9, 0.31+h, 267.15, -0.0027, 0.25)
    p2, run2 = f2.Heston_fft(1.5, 8, 267.15*2.7, 275)
    vega = (p2 - p1) / (2 * h)
    
    
    
    return vega    
      
def vega_BS() :
    sigma = 0.18820324096679686
    h = 0.001
    p1 = Eurocall(267.15, 275, 0.25, sigma+h, -0.0027).BScall()
    p2 = Eurocall(267.15, 275, 0.25, sigma-h, -0.0027).BScall()
    
    vega = (p1 - p2) / (2 * h)
    
    return vega
    
def sol() :
    sol = minimize(mini, np.array([0.15, 0.09, 0.1, 0.86, 0.48]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    sol2 = minimize(mini, np.array([0.10, 0.05, 0.05, 0.6, 0.3]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    sol3 = minimize(mini, np.array([0.21, 0.08, 0.08, -0.3, 0.2]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    sol4 = minimize(mini, np.array([0.15, 0.09, 0.1, 0.86, 0.48]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))
    sol5 = minimize(mini, np.array([0.10, 0.05, 0.05, 0.6, 0.3]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))

    sol1 = minimize(dmini, np.array([0.08, 0.0001, 0.009, 0.6, 0.5]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    sol6 = minimize(dmini, np.array([0.10, 0.05, 0.01, 0.6, 0.3]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    sol7 = minimize(dmini, np.array([0.21, 0.3, 0.4, -0.3, 0.2]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    sol8 = minimize(dmini, np.array([0.15, 0.09, 0.1, 0.86, 0.48]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))
    sol9 = minimize(dmini, np.array([0.10, 0.06, 0.05, 0.6, 0.3]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))

    return sol,sol2,sol3,sol4,sol5,sol1,sol6,sol7,sol8,sol9  
        
    










if __name__ == "__main__" :
    
    a = table()
    v = [0.3225, 0.2836, 0.2473, 0.2178, 0.2021, 0.1818, 0.1824, 0.1645, 0.1574, 0.1462, 0.1370, 0.1256, 0.1148, 0.1094]
    v = np.array(v).reshape((7, 2))
    k_2 = [a[i][0] for i in range(len(a))]
    k_3 = [a[i][1] for i in range(len(a))]
    v_2 = [v[i][0] for i in range(len(v))]
    v_3 = [v[i][1] for i in range(len(v))]
    
    p_1 = vol_func(k_2, v_2)
    p_2 = vol_func(k_3, v_3)
    f_1 = np.poly1d(p_1)
    f_2 = np.poly1d(p_2)
    
    
    risk_den(f_1, 1/12)
    risk_den(f_2, 0.25)
    
    risk_den1(0.1824, 1/12)
    risk_den1(0.1645, 0.25)
    b = price(f_1, 70, 110, 1/12)
    c = price(f_2, 105, 125, 0.25)
    
    v_4 = [(v_2[i] + v_3[i]) / 2 for i in range(len(v_2))]
    
    delta = [0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1]
    
    k_4 = [Kcall(v_4[i], 1/6, delta[i]) for i in range(7)]
    
    p_3 = vol_func(k_4, v_4)
    
    f_3 = np.poly1d(p_3)
    
    d = price1(f_3, 100, 125, 1/6) 
    df = pd.read_excel('mf796-hw5-opt-data.xlsx')
    df['call_mid'] = (df['call_bid'] + df['call_ask']) / 2
    df['put_mid'] = (df['put_bid'] + df['put_ask']) / 2
    df1 = df.copy()
    plt.plot(df['call_mid'])
    plt.ylabel('call_mid')
    plt.xlabel('number')
    plt.show()
    plt.plot(df['put_mid'])
    plt.ylabel('put_mid')
    plt.xlabel('number')
    plt.show()
    df2 = df1[df1['expDays'] == 49].copy()
    df3 = df1[df1['expDays'] == 140].copy()
    df4 = df1[df1['expDays'] == 203].copy()
    df2['c_change'] = (df2['call_mid'] - df2['call_mid'].shift(1)) / 5
    df2['p_change'] = (df2['put_mid'] - df2['put_mid'].shift(1)) / 5
    plt.plot(df2[['c_change','p_change']])
    plt.ylabel('price_change_rate')
    plt.xlabel('number')
    plt.show()
    
    df3['c_change'] = (df3['call_mid'] - df3['call_mid'].shift(1)) / 5
    df3['p_change'] = (df3['put_mid'] - df3['put_mid'].shift(1)) / 5
    plt.plot(df3[['c_change','p_change']])
    plt.ylabel('price_change_rate')
    plt.xlabel('number')
    plt.show()
    
    df4['c_change'] = (df4['call_mid'] - df4['call_mid'].shift(1)) / 5
    df4['p_change'] = (df4['put_mid'] - df4['put_mid'].shift(1)) / 5
    plt.plot(df4[['c_change','p_change']])
    plt.ylabel('price_change_rate')
    plt.xlabel('number')
    plt.show()
    
    
    parameter = np.array(df1[['expT','K','call_mid']].copy()).reshape(44,3)
    df1['spread'] = 1/(df1['call_ask'] - df1['call_bid'])
    parameter = np.c_[parameter, np.array([df1['spread']]).reshape(44,1)]
    # sol = minimize(mini, np.array([0.15, 0.09, 0.1, 0.86, 0.48]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    # sol2 = minimize(mini, np.array([0.10, 0.05, 0.05, 0.6, 0.3]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    # sol3 = minimize(mini, np.array([0.21, 0.08, 0.08, -0.3, 0.2]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    # sol4 = minimize(mini, np.array([0.15, 0.09, 0.1, 0.86, 0.48]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))
    # sol5 = minimize(mini, np.array([0.10, 0.05, 0.05, 0.6, 0.3]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))

    # sol1 = minimize(dmini, np.array([0.08, 0.0001, 0.009, 0.6, 0.5]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    # sol6 = minimize(dmini, np.array([0.10, 0.05, 0.01, 0.6, 0.3]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    # sol7 = minimize(dmini, np.array([0.21, 0.3, 0.4, -0.3, 0.2]), method="TNC", bounds=((0,1),(0,1),(0,1),(-1,1), (0,1)))
    # sol8 = minimize(dmini, np.array([0.15, 0.09, 0.1, 0.86, 0.48]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))
    # sol9 = minimize(dmini, np.array([0.10, 0.06, 0.05, 0.6, 0.3]), method="TNC", bounds=((0,0.5),(0,0.8),(0,2),(-0.8,1), (0,1)))










    
