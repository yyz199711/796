#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:52:39 2020

@author: Yuyang Zhang
"""

import scipy.stats as si
import numpy as np
from scipy.interpolate import splrep, splev

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
    
        call = self.s * si.norm.cdf(self.d_1()) - self.k * np.exp(-self.r * self.t) * si.norm.cdf(self.d_2())
    
        return call

def sigma_BS(price, k) :
    upper_bound = 1
    lower_bound = 0.0001
    guess = 0.5
    a = Eurocall(312.86, k, 0.578, guess, 0.0071).BScall()
    # print(b)
    while abs(a - price) > 0.0001 :
        if a > price :
            upper_bound = guess
            guess = (upper_bound + lower_bound) / 2
            
        if a < price :
            lower_bound = guess
            guess = (lower_bound + upper_bound) / 2
                
            
            
        a = Eurocall(312.86, k, 0.578, guess, 0.0071).BScall()
            # print(guess, a)
        
    # d1 = Eurocall(267.15, 275, 0.25, guess, -0.0027).d_1()
    # print(Eurocall(267.15, 275, 0.25, guess, -0.0027).BScall())
    
    return guess

def engin_A():
    r = 0.71 / 100
    sigma = 0.122
    M = 50
    T = 0.578
    N = 50
    S_max = 500
    h_s = S_max / M
    h_t = T / N
    t = h_t * np.linspace(0, N, N+1)
    s = h_s * np.arange(1, M, 1)
    A = np.zeros((M-1, M-1))
    u_end = sigma ** 2 * s[M - 2] ** 2 * h_t / (2 * h_s ** 2) + r * s[M - 2] * h_t / (2 * h_s)
    for i in range(1, M) :
        a_i = 1 - sigma ** 2 * s[i - 1] ** 2 * h_t / h_s ** 2 - r * h_t
        l_i = sigma ** 2 * s[i - 1] ** 2 * h_t / (2 * h_s ** 2) - r * s[i - 1] * h_t / (2 * h_s)
        u_i = sigma ** 2 * s[i - 1] ** 2 * h_t / (2 * h_s ** 2) + r * s[i - 1] * h_t / (2 * h_s)
        
        A[i - 1][i - 1] = a_i
        if i > 1 :
            A[i - 1][i - 2] = l_i
        if i < M - 1 :
            A[i - 1][i] = u_i
    
    w, v = np.linalg.eig(A)
    flag = 0
    index = []
    for i in range(len(w)) :
        if abs(w[i]) < 1 :
            flag += 0
        else :
            index += [i]
            flag += 1
    return A, w, flag, index, u_end
            
def cs_price():
    k_1 = 315
    k_2 = 320
    A, w, flag, inde, u_end = engin_A()
    r = 0.71 / 100
    sigma = 0.122
    M = 50
    T = 0.578
    N = 50
    S_max = 500
    # S_min = 200
    h_s = S_max / M
    h_t = T / N
    t = h_t * np.linspace(0, N, N+1)
    s = h_s * np.arange(1, M, 1)
    A_N = np.linalg.matrix_power(np.matrix(A), len(t)-1)
    # print(A_N)
    c_N = np.zeros((M-1, 1))
    
    for i in range(M-1) :
        c_N[i][0] = max(s[i] - k_1, 0) - max(s[i] - k_2, 0)
    
    c_N = np.matrix(c_N)
    
    add = A_N * c_N
        
    
    for j in range(1, len(t)) :
        
        b_j = np.zeros((M-1, 1))
        b_j[M-2][0] = np.exp(-r * (T - t[j])) * (k_2 - k_1) * u_end
        b_j = np.matrix(b_j)
        if j == 1 :
            coef = np.matrix(np.identity(M-1))
        else :
            coef = np.linalg.matrix_power(np.matrix(A), j - 1)
        add += coef * b_j
        # print(coef * b_j)
    
    return add, s

def c_price(s_t) :
    d, s = cs_price()
    tck = splrep(s, d)
    p = splev(s_t, tck)
    
    return float(p)
    
   
def acs_price():
    k_1 = 315
    k_2 = 320
    # A, w, flag, inde, u_end = engin_A()
    r = 0.71 / 100
    sigma = 0.122
    M = 50
    T = 0.578
    N = 50
    S_max = 500
    # S_min = 200
    h_s = S_max / M
    h_t = T / N
    # t = h_t * np.linspace(0, N, N+1)
    s = [h_s * i for i in range(M+1)]
    # print(s)
    call_price = np.zeros((M+1, N+1))
    # print(call_price[0][0])
    for j in range(N+1) :
        call_price[0][j] = (k_2 - k_1)
    
    for i in range(M+1) :
        call_price[i][N] = max(s[M-i] - k_1, 0) - max(s[M-i] - k_2, 0)
        # print(call_price[i][N])
    
    a = np.zeros(M-1)
    l = np.zeros(M-1)
    u = np.zeros(M-1)
    for k in range(M-1) :
        a[k] = 1 - sigma ** 2 * s[k+1] ** 2 * h_t / h_s ** 2 - r * h_t
        l[k] = sigma ** 2 * s[k+1] ** 2 * h_t / (2 * h_s ** 2) - r * s[k+1] * h_t / (2 * h_s)
        u[k] = sigma ** 2 * s[k+1] ** 2 * h_t / (2 * h_s ** 2) + r * s[k+1] * h_t / (2 * h_s)
    # print(a, l, u)
    # print(a)
    # count = 0  
    for q in range(1, M) :
        for b in range(N) :
            
            call = a[M-q-1] * call_price[q][N-b] + l[M-q-1] * call_price[q+1][N-b] + u[M-1-q] * call_price[q-1][N-b]
            
                            
            compare = max(s[M-q] - k_1, 0) - max(s[M-q] - k_2, 0)
            # compare = 0
            
            call_price[q][N-b-1] = max(compare, call)
            # if b == 0 and q == 37:
            #     print(call)
            #     print(compare)
            #     print(call_price[q-1][N-b], u[M-q-1])
                # count += 1
    # print(count)
    
    call_spread = np.array([call_price[M-d][0] for d in range(M+1)])
    
    return call_spread, s
        

        

        
    
    
    
    
    
    
    # A_N = np.linalg.matrix_power(np.matrix(A), len(t)-1)
    # # print(A_N)
    # c_N = np.zeros((M-1, 1))
    
    # for i in range(M-1) :
    #     c_N[i][0] = max(s[i] - k_1, 0) - max(s[i] - k_2, 0)
    
    # c_N = np.matrix(c_N)
    
    # add = A_N * c_N
        
    
    # for j in range(1, len(t)) :
        
    #     b_j = np.zeros((M-1, 1))
    #     b_j[M-2][0] = (k_2 - k_1) * u_end
    #     b_j = np.matrix(b_j)
    #     if j == 1 :
    #         coef = np.matrix(np.identity(M-1))
    #     else :
    #         coef = np.linalg.matrix_power(np.matrix(A), j - 1)
    #     add += coef * b_j
    #     # print(coef * b_j)
    
    # return add, s

def ac_price(s_t) :
    d, s = acs_price()
    tck = splrep(s, d)
    p = splev(s_t, tck)
    
    return float(p)
    
    
        
        
        
         
        
















if __name__ == "__main__" :
    a = sigma_BS(8.99, 320)
    # b = sigma_BS(12, 315)
    c = engin_A()
    d = cs_price()
    e = c_price(312.86)
    f = acs_price()
    g = ac_price(312.86)

















