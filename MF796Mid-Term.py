#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:34:45 2020

@author: Yuyang Zhang
"""
import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
#from scipy.optimize import brentq as root
from scipy.optimize import root
from scipy import interpolate
import pandas as pd
from scipy.optimize import minimize
from cvxopt import solvers, matrix


''' This function is for Problem 3(b) simulate the Heston model and get the price.'''
def Sim_Heston_Asian(x):
    K = 100 # strike
    k = 1  # kappa
    xi = 0.25 
    nu_0 = 0.05 ** 2 # variance
    rho = -0.5
    theta = 0.1
    S_0 = 100
    r = 0
    q = 0 # dividend rate
    T = 0.25
    cov = np.array([[1, rho], [rho, 1]]) # covariance matrix for standard normal
    mean = np.array([0, 0]) 
    np.random.seed(1)
    N = x
    N_1 = T * 252 # number for T
    delta_T = 1 / 252
    Sim_T = np.zeros(N)
    for i in range(N) :
        S = np.zeros(int(N_1) + 1)
        nu = np.zeros(int(N_1) + 1)
        nu[0] = nu_0
        S[0] = S_0
        Z = np.random.multivariate_normal(mean, cov, int(N_1))
        for j in range(1, int(N_1) + 1) :
            S[j] = S[j-1] + (r-q) * S[j-1] * delta_T + np.sqrt(nu[j-1]) * S[j-1] * np.sqrt(delta_T) * Z[j-1][0]
            nu[j] = nu[j-1] + k * (theta - nu[j-1]) * delta_T + xi * np.sqrt(nu[j-1]) * np.sqrt(delta_T) * Z[j-1][1]
            # the Euler discretization
            if nu[j] <= 0 :
                nu[j] = 0 # truncate the value
        Sim_T[i] = np.exp(-r*T) * max(np.mean(S) - K, 0)
    
    return np.mean(Sim_T)

'''This function is for Problem 3(c) and plot the theoretical and practical convergence rate'''

def plot_convergence() :
    N = np.linspace(100, 10000, 50) # simulated paths
    
    a = Sim_Heston_Asian(100)
    p_ratio = np.zeros(50)
    the_con = np.zeros(50)
    c = (1/100) ** 0.5 # start point for N^{-1/2}
    for i in range(50) :
        p = Sim_Heston_Asian(int(N[i]))
        p_ratio[i] = abs(p - 0.995) / a
        print(p)
        the_con[i] = abs(a - 0.995) * (1/ N[i]) ** 0.5 / (c * a)  # theoretical convergence rate
        
    plt.plot(N, p_ratio)
    plt.plot(N, the_con)
    plt.xlabel('N')
    plt.ylabel('Convergence')
    plt.legend(['Practical', 'Theoretical'])
    plt.show()
        
''' This function is for Problem 3(f) simulate the Heston model by control variate and get the price.'''        
        
def Sim_Heston_Asian_Control(x):
    K = 100 # strile
    k = 1  # kappa
    xi = 0.25 
    nu_0 = 0.05 ** 2 # variance
    rho = -0.5
    theta = 0.1
    S_0 = 100
    r = 0
    q = 0 # dividend rate
    T = 0.25
    cov = np.array([[1, rho], [rho, 1]]) # covariance matrix for standard normal
    mean = np.array([0, 0]) 
    np.random.seed(1)
    N = x
    N_1 = T * 252 # number for T
    delta_T = 1 / 252
    Sim_T_1 = np.zeros(N)
    Sim_T_2 = np.zeros(N)
    for i in range(N) :
        S = np.zeros(int(N_1) + 1)
        nu = np.zeros(int(N_1) + 1)
        nu[0] = nu_0
        S[0] = S_0
        Z = np.random.multivariate_normal(mean, cov, int(N_1))
        for j in range(1, int(N_1) + 1) :
            S[j] = S[j-1] + (r-q) * S[j-1] * delta_T + np.sqrt(nu[j-1]) * S[j-1] * np.sqrt(delta_T) * Z[j-1][0]
            nu[j] = nu[j-1] + k * (theta - nu[j-1]) * delta_T + xi * np.sqrt(nu[j-1]) * np.sqrt(delta_T) * Z[j-1][1]
            # the Euler discretization
            if nu[j] <= 0 :
                nu[j] = 0 # truncate the value
        Sim_T_1[i] = np.exp(-r*T) * max(np.mean(S) - K, 0) #Asian
        Sim_T_2[i] = np.exp(-r*T) * max(S[-1] - K, 0) #European
        Sim_T = np.append(Sim_T_1, Sim_T_2, axis=0).reshape((2, N))
        cov_matrix = np.cov(Sim_T)
        c = -cov_matrix[0][1] / cov_matrix[1][1] # the parameter for control variate
    
    return c, np.mean(Sim_T_2)
 
''' choose c = -0.34531223549086243, the European Call Price is 2.238790757354707'''
def Sim_Heston_Asian_Control_1(x):
    c = -0.34531223549086243
    call_Euro = 2.238790757354707
    K = 100 # strile
    k = 1  # kappa
    xi = 0.25 
    nu_0 = 0.05 ** 2 # variance
    rho = -0.5
    theta = 0.1
    S_0 = 100
    r = 0
    q = 0 # dividend rate
    T = 0.25
    cov = np.array([[1, rho], [rho, 1]]) # covariance matrix for standard normal
    mean = np.array([0, 0]) 
    np.random.seed(1)
    N = x
    N_1 = T * 252 # number for T
    delta_T = 1 / 252
    Sim_T_1 = np.zeros(N)
    Sim_T_2 = np.zeros(N)
    for i in range(N) :
        S = np.zeros(int(N_1) + 1)
        nu = np.zeros(int(N_1) + 1)
        nu[0] = nu_0
        S[0] = S_0
        Z = np.random.multivariate_normal(mean, cov, int(N_1))
        for j in range(1, int(N_1) + 1) :
            S[j] = S[j-1] + (r-q) * S[j-1] * delta_T + np.sqrt(nu[j-1]) * S[j-1] * np.sqrt(delta_T) * Z[j-1][0]
            nu[j] = nu[j-1] + k * (theta - nu[j-1]) * delta_T + xi * np.sqrt(nu[j-1]) * np.sqrt(delta_T) * Z[j-1][1]
            # the Euler discretization
            if nu[j] <= 0 :
                nu[j] = 0 # truncate the value
        Sim_T_1[i] = np.exp(-r*T) * max(np.mean(S) - K, 0) #Asian
        Sim_T_2[i] = np.exp(-r*T) * max(S[-1] - K, 0) #European
        Sim_T = Sim_T_1 + c * (Sim_T_2 - call_Euro)
    
    return np.mean(Sim_T)
''' This function is to plot the convergence rate for control variate'''           
def plot_convergence_control() :
    N = np.linspace(100, 10000, 50) # simulated paths
    
    a = Sim_Heston_Asian(100)
    a_1 = Sim_Heston_Asian_Control_1(100)
    p_ratio = np.zeros(50)
    p_ratio_wcv = np.zeros(50)
    the_con = np.zeros(50)
    c = (1/100) ** 0.5 # start point for N^{-1/2}
    for i in range(50) :
        p = Sim_Heston_Asian(int(N[i]))  # Pratical convergence rate without control variate
        p_1 = Sim_Heston_Asian_Control_1(int(N[i])) # Pratical convergence rate with control variate
        
        
        
        p_ratio[i] = abs((p - 0.995) / a)
        p_ratio_wcv[i] = (abs(p_1 - 0.995) / a_1) * (abs(a - 0.995) / a) / (abs(a_1 - 0.995) / a_1) 
        print(p_1)
        the_con[i] = abs(a - 0.995) * (1/ N[i]) ** 0.5 / (c * a)  # theoretical convergence rate
        
    plt.plot(N, p_ratio)
    plt.plot(N, p_ratio_wcv)
    plt.plot(N, the_con)
    plt.xlabel('N')
    plt.ylabel('Convergence')
    plt.legend(['Practical without Control','Practical with Control','Theoretical'])
    plt.show()




''' This is for Problem4(a)'''
ret = pd.read_csv('DataForProblem4.csv')
symbols = [symbol for symbol in ret.columns]
ret = ret.drop('Unnamed: 11', axis = 1)
ret_a = ret.loc[:, 'Sec1' : 'Sec10']
cov = matrix(np.cov(ret_a.T))
q_1 = matrix(np.zeros((10, 1)))
G = matrix(-np.identity(10))
h = matrix(np.zeros((10,1)))
A = matrix(np.ones((1, 10)))
b = matrix([1.0])

sol1 = solvers.qp(cov,q_1, G=G, h=h, A=A, b=b)
print(sol1['x'])


'''This is for Problem4(b)'''
ex_return = np.array(ret_a.mean()).reshape((10, 1))
q_2 = matrix(-ex_return)
sol2 = solvers.qp(cov,q_2, G=G, h=h, A=A, b=b)
print(sol2['x'])




'''This is for Problem4(d)'''

B = np.matrix(ret.loc[:, 'B1']).T
q_3 = matrix(-np.matrix(ret_a).T * B)
cov_1 = matrix(np.matrix(ret_a).T * np.matrix(ret_a))

    
sol3 = solvers.qp(cov_1,q_3, G=G, h=h, A=A, b=b)
print(sol3['x'])

'''This is for Problem4(f)'''
ret_b = ret_a.copy()
ret_b['Cash Position'] = 0
ret_b = 1 + ret_b
ret_b = np.matrix(ret_b)

def ex_u(x) :
    wealth = ret_b * np.matrix(x).T
    
    log_wealth = np.log(wealth)
    
    return -np.mean(log_wealth)

x_1 = np.ones((1, 11)) / 11
bds_1 = ((0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1))
cons_1 = ({'type': 'eq', 'fun' : lambda x : np.sum(x) - 1})
sol4 = minimize(ex_u, x_1, method = 'SLSQP', bounds = bds_1, constraints = cons_1)







# if __name__ == "__main__" :
#     a = Sim_Heston_Asian(10000)
#     c = Sim_Heston_Asian_Control(10000)
#     plot_convergence()
#     plot_convergence_control()








