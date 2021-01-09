#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 02:49:46 2020

@author: shousakai
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

class BSOption:
    """ encapsulates the data required to do Black-Scholes option pricing formula.
    """
    def __init__(self,s,x,t,sigma,rf,div):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        
    def __repr__(self):
        """ returns a string representation of the BSOption object
        """
        s = 's = $%.2f, x = $%.2f, t = %.2f(years), sigma = %.3f, rf = %.3f, div = %.2f' % (self.s, self.x, self.t, self.sigma,self.rf, self.div)
        return s
    
    def d1(self):
        """ calculates d1 of the option
        """
        numerator = math.log(self.s/self.x) + (self.rf-self.div+self.sigma**2*0.5)*self.t      # Numerator of d1
        denominator = self.sigma * self.t**0.5                                                 # Denominator of d1
        
        return numerator/denominator
        
    def d2(self):
        """ calculates d2 of the option
        """
        return self.d1() - self.sigma*self.t**0.5
    
    def nd1(self):
        """ calculates N(d1) of the option
        """
        return norm.cdf(self.d1())
    
    def nd2(self):
        """ calculates N(d2) of the option
        """
        return norm.cdf(self.d2())
    
class BSEuroCallOption(BSOption):
    def __repr__(self):
        """ returns a string representation of the BSEuroCallOption object
        """
        s = 'BSEuroCallOption, value = $%.2f, \n' % self.value()
        s += 'parameters = (' + BSOption.__repr__(self) + ')'
        
        return s
    
    def value(self):
        """ calculates value for the option
        """ 
        c = self.nd1() * self.s * math.exp(-self.div * self.t)
        c -= self.nd2() * self.x * math.exp(-self.rf * self.t)
        
        return c
    
    def delta(self):
        """ calculates delta for the option
        """
        return self.nd1()

class FTT:
    def __init__(self,sigma,eta0,kappa,rho,theta,S0,r,T):
        self.sigma = sigma
        self.eta0 = eta0
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        self.S0 = S0
        self.r = r
        self.T = T
        
    def Heston_fft(self,alpha,n,B,K):
        """ Define a function that performs fft on Heston process
        """
        bt = time.time()
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        Eta = B / N
        Lambda_Eta = 2 * math.pi / N
        Lambda = Lambda_Eta / Eta
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * Eta
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - Lambda * N / 2
        km = Beta + (m-1) * Lambda
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Heston_cf(u)
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        # Compute FTT
        xx = (Eta/2) * Psi_vj * np.exp(-ii * Beta * vj) * (2 - self.dirac(J-1))
        zz = np.fft.fft(xx)
        
        # Option price
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        zz2 = Mul * np.array(zz).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * Lambda)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [zz2[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        price =  np.exp(-r*T)*interpolate.splev(K, tck).real
        et = time.time()
        
        runt = et-bt

        return(price,runt)
    
    def dirac(self,n):
        """ Define a dirac delta function
        """
        y = np.zeros(len(n),dtype = complex)
        y[n==0] = 1
        return y
        
    def Heston_cf(self,u):
        """ Define a function that computes the characteristic function for variance gamma
        """
        sigma = self.sigma
        eta0 = self.eta0
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        T = self.T
        
        ii = complex(0,1)
        
        l = cmath.sqrt(sigma**2*(u**2+ii*u)+(kappa-ii*rho*sigma*u)**2)
        w = np.exp(ii*u*np.log(S0)+ii*u*(r-0)*T+kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma**2)/(cmath.cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*cmath.sinh(l*T/2))**(2*kappa*theta/sigma**2)
        y = w*np.exp(-(u**2+ii*u)*eta0/(l/cmath.tanh(l*T/2)+kappa-ii*rho*sigma*u))
        
        return y
    
    def alpha_plot(self,lst,n,B,K):
        yy = np.array([self.Heston_fft(a,n,B,K)[0] for a in lst])
        
        plt.plot(lst,yy)
        plt.title("Fig. FFT European Call Option Price vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("FFT European Call Option Price")
        plt.show()
        
    def NB_plot(self,n_list,B_list,K):
        zz = np.zeros((len(n_list),len(B_list)))
        ee = np.zeros((len(n_list),len(B_list)))
        cc = []
        xx, yy = np.meshgrid(n_list, B_list)
        for i in range(len(n_list)):
            for j in range(len(B_list)):
                temp = self.Heston_fft(1,n_list[i],B_list[j],K)
                zz[i][j] = temp[0]
                ee[i][j] = 1/((temp[0]-21.27)**2*temp[1])
                cc += [(ee[i][j],n_list[i],B_list[j])]
                
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, zz.T, rstride=1, cstride=1, cmap='rainbow')
        plt.title("Fig. FFT European Call Option Price vs N & B")
        ax.set_xlabel("N")
        ax.set_ylabel("B")
        ax.set_zlabel("FFT European Call Option Price")
        plt.show()
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, ee.T, rstride=1, cstride=1, cmap='rainbow')
        plt.title("Fig. FFT Efficiency vs N & B")
        ax.set_xlabel("N")
        ax.set_ylabel("B")
        ax.set_zlabel("FFT Efficiency")
        plt.show()
        
        print(max(cc))
        
        
        
def plot_vol_K(price_list,K_list):
    vol = []
    
    for i in range(len(K_list)):
        result = root(lambda x: BSEuroCallOption(150, K_list[i], 0.25, x, 0.025, 0.00).value()-price_list[i],0.3)
        vol += [result.x]
        
    vol = np.array(vol)
    plt.plot(K_list,vol)
    plt.title("Fig. Implied Volatility vs Strike K")
    plt.xlabel("Strike K")
    plt.ylabel("Implied Volatility")
    plt.show()
    
def plot_vol_T(price_list,t_list):
    vol = []
    
    for i in range(len(t_list)):
        result = root(lambda x: BSEuroCallOption(150, 150, t_list[i], x, 0.025, 0.00).value()-price_list[i],0.3)
        vol += [result.x]
        
    vol = np.array(vol)
    plt.plot(t_list,vol)
    plt.title("Fig. Implied Volatility vs Expiry T")
    plt.xlabel("Expiry T")
    plt.ylabel("Implied Volatility")
    plt.show()
    
def Sim_Heston_Euro():
    K = 285
    k = 3.51
    sigma = 1.17
    nu_0 = 0.034
    rho = -0.77
    theta = 0.052
    S_0 = 282
    r = 1.5 / 100
    q = 1.77 / 100
    T = 1
    cov = np.array([[1, rho], [rho, 1]])
    mean = np.array([0, 0])
    np.random.seed(2)
    N = 1000
    N_1 = 1000
    delta_T = T / N_1
    Sim_T = np.zeros(N)
    for i in range(N) :
        S = np.zeros(N_1 + 1)
        nu = np.zeros(N_1 + 1)
        nu[0] = nu_0
        S[0] = S_0
        Z = np.random.multivariate_normal(mean, cov, N_1)
        for j in range(1, N_1 + 1) :
            S[j] = S[j-1] + (r-q) * S[j-1] * delta_T + np.sqrt(nu[j-1]) * S[j-1] * np.sqrt(delta_T) * Z[j-1][0]
            nu[j] = nu[j-1] + k * (theta - nu[j-1]) * delta_T + sigma * np.sqrt(nu[j-1]) * np.sqrt(delta_T) * Z[j-1][1]
            if nu[j] <= 0 :
                nu[j] = 0
        Sim_T[i] = np.exp(-r*T) * max(S[-1] - K, 0)
    
    return np.mean(Sim_T)

def Sim_Heston_Up(x) :
    K_1 = 285
    K_2 = 315
    k = 3.51
    sigma = 1.17
    nu_0 = 0.034
    rho = -0.77
    theta = 0.052
    S_0 = 282
    r = 1.5 / 100
    q = 1.77 / 100
    T = 1
    cov = np.array([[1, rho], [rho, 1]])
    mean = np.array([0, 0])
    np.random.seed(2)
    N = x
    N_1 = 1000
    delta_T = T / N_1
    Sim_T_1 = np.zeros(N)
    Sim_T_2 = np.zeros(N)
    for i in range(N) :
        S = np.zeros(N_1 + 1)
        nu = np.zeros(N_1 + 1)
        nu[0] = nu_0
        S[0] = S_0
        Z = np.random.multivariate_normal(mean, cov, N_1)
        for j in range(1, N_1 + 1) :
            S[j] = S[j-1] + (r-q) * S[j-1] * delta_T + np.sqrt(nu[j-1]) * S[j-1] * np.sqrt(delta_T) * Z[j-1][0]
            nu[j] = nu[j-1] + k * (theta - nu[j-1]) * delta_T + sigma * np.sqrt(nu[j-1]) * np.sqrt(delta_T) * Z[j-1][1]
            if nu[j] <= 0 :
                nu[j] = 0
        Sim_T_2[i] = np.exp(-r*T) * max(S[-1] - K_1, 0)
        if max(S) >= K_2 :
            Sim_T_1[i] = 0
        else :
            
            Sim_T_1[i] = max(S[-1] - K_1, 0)
    
    Sim_T = np.append(Sim_T_1, Sim_T_2).reshape((2, N))
    cov_matrix = np.cov(Sim_T)
    c = -cov_matrix[0][1] / cov_matrix[1][1]
    print(c)
    Sim_T_3 = Sim_T_1 + c * (Sim_T_2 - 17.990776751230484)
    
    
    
    
    return np.mean(Sim_T_1), np.mean(Sim_T_3)

def cal_price() :
    p = np.zeros(50)
    p_1 = np.zeros(50)
    for i in range(1, 51) :
        x = int(i * 10)
        p[i-1], p_1[i-1] = Sim_Heston_Up(x)
        print(p[i-1])
    n = np.linspace(10, 500, 50)
    plt.plot(n, p)
    plt.plot(n, p_1)

    plt.xlabel('N')
    plt.ylabel('Price')
    plt.legend(['Price without Control', 'Price with Control'])
    
    
    
    

    
    return p
        
    
if __name__ == "__main__" :
    a = FTT(1.17, 0.034, 3.51, -0.77, 0.052, 282, (1.5-1.77)/100, 1)
    print(a.Heston_fft(1.5, 11, 800, 285))
    

































        
