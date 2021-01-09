#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 08:43:32 2020

@author: shousakai
"""
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt


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
    
    def den(self, x) :
        density = 1 / (np.sqrt(2 * np.pi)) * np.exp(-np.power(x, 2) / 2)
        # print(density)
        return density
        
    
    def lrm(self, n, a) :
        w_1 = (self.d_1() - a) / n
        w_2 = (self.d_2() - a) / n
        l_1 = 0
        l_2 = 0
        for i in range(n) :
            x_1 = a + i * w_1
            x_2 = a + i * w_2
            l_1 += self.den(x_1) * w_1
            l_2 += self.den(x_2) * w_2
        
        c = self.s * l_1 - np.exp(-self.r * self.t) * self.k * l_2    
        
        return c
    
    
    
    def lrmerr(self, n, a) :
        
        return self.lrm(n, a) - self.BScall()
    
    def mrm(self, n, a) :
        w_1 = (self.d_1() - a) / n
        w_2 = (self.d_2( )- a) / n
        l_1 = 0
        l_2 = 0
        for i in range(n) :
            x_1 = a + i * w_1 + 0.5 * w_1
            x_2 = a + i * w_2 + 0.5 * w_2
            l_1 += self.den(x_1) * w_1
            l_2 += self.den(x_2) * w_2
            
        c = self.s * l_1 - np.exp(-self.r * self.t) * self.k * l_2  
        
        return c
        
    
    def mrmerr(self, n, a) :
        
        return self.mrm(n, a) - self.BScall()
    
    def Gauss(self, n, a) :
        w = np.polynomial.legendre.leggauss(n)[1]
        x = np.polynomial.legendre.leggauss(n)[0] 
        l_1 = 0
        l_2 = 0
        for i in range(len(x)) :
            x_1 = (x[i] + 1) * (self.d_1() - a) / 2 + a
            x_2 = (x[i] + 1) * (self.d_2() - a) / 2 + a
            wi_1 = (self.d_1() - a) / 2
            wi_2 = (self.d_2() - a) / 2
            l_1 += wi_1 * w[i] * self.den(x_1)
            l_2 += wi_2 * w[i] * self.den(x_2)
        
        c = self.s * l_1 - np.exp(-self.r * self.t) * self.k * l_2 
            

        return c
        
    def Gausserr(self, n, a) :
        
        return self.Gauss(n, a) - self.BScall()
    
    def plotlrm(self, a) :
        y = np.array([abs(self.lrmerr(i, a)) for i in range(5, 101)])
        x_1 = np.array([1 / i * 5 * abs(self.lrmerr(5, a)) for i in range(5, 101)])
        x_2 = np.array([(5 / i) ** 2 * abs(self.lrmerr(5, a)) for i in range(5, 101)])
        x_3 = np.array([(5 / i) ** 3 * abs(self.lrmerr(5, a)) for i in range(5, 101)])
        plt.plot(y)
        plt.plot(x_1)
        plt.plot(x_2)
        plt.plot(x_3)
        plt.legend(['Left-Riemann', '1/N', '1/N^2', '1/N^3'])
        
    def plotmrm(self, a) :
        y = np.array([abs(self.mrmerr(i, a)) for i in range(5, 101)])
        # x_1 = np.array([1/ i for i in range(5, 101)])
        x_2 = np.array([(5 / i) ** 2 * abs(self.mrmerr(5, a)) for i in range(5, 101)])
        x_3 = np.array([(5 / i) ** 3 * abs(self.mrmerr(5, a)) for i in range(5, 101)])
        plt.plot(y)
        # plt.plot(x_1)
        plt.plot(x_2)
        plt.plot(x_3)
        plt.legend(['Mid-rule', '1/N^2', '1/N^3'])
        
    def plotGauss(self, a) :
        y = np.array([abs(self.Gausserr(i, a)) for i in range(5, 101)])
        # x_1 = np.array([1/ i for i in range(5, 101)])
        x_2 = np.array([i ** (-2 * i) * 5 ** 10 * abs(self.Gausserr(5, a))  for i in range(5, 101)])
        # x_3 = np.array([1 / i ** 3 for i in range(5, 101)])
        plt.plot(y)
        # plt.plot(x_1)
        plt.plot(x_2)
        # plt.plot(x_3)
        plt.legend(['Gauss nodes', 'N^{-2N}'])   
        
            
        
            

class JNormal :
    def __init__(self, s, sigma_1, sigma_2, rho, r, t_1, t_2, k) :
        self.s = s
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.rho = rho
        self.r = r
        self.t_1 = t_1
        self.t_2 = t_2
        self.k = k
        self.mu_1 = self.s * np.exp(self.t_1 * self.r)
        self.mu_2 = self.s * np.exp(self.t_2 * self.r)
        
    def p_1(self, x) :
        
        p_1 = 1 / (self.sigma_1 * np.sqrt(2 * np.pi)) * np.exp(-np.power(x - self.mu_1 , 2) / (2 * np.power(self.sigma_1, 2)))
        # print(p_1)
        return p_1 * max(x - self.k, 0)
    
    def p_2(self, x) :
        
        p_2 = 1 / (self.sigma_2 * np.sqrt(2 * np.pi)) * np.exp(-np.power(x - self.mu_2 , 2) / (2 * np.power(self.sigma_2, 2)))
        
        return p_2
    
    def p(self, x, y) :
        
        z = np.power(x - self.mu_1, 2) / np.power(self.sigma_1, 2) - 2 * self.rho * (x - self.mu_1) * (y - self.mu_2) / (self.sigma_1 * self.sigma_2) + np.power(y - self.mu_2, 2) / np.power(self.sigma_2, 2)                  
        
        p = 1 / (2 * np.pi * self.sigma_1 * self.sigma_2 * np.sqrt(1 - self.rho ** 2)) * np.exp(- z / (2 * (1 - np.power(self.rho, 2))))
                                                                                                
        return p                                                                                        
    
    def sin_value(self) :
        x = np.polynomial.legendre.leggauss(100)[0]
        w = np.polynomial.legendre.leggauss(100)[1]
        l = 0
        for i in range(len(x)) :
            y = (x[i] + 1) * (self.mu_1 + 10 * self.sigma_1 - self.k) / 2 + self.k
            f = self.p_1(y) * w[i] * (self.mu_1 + 10 * self.sigma_1 - self.k) / 2 
            l += f
        
        l *= np.exp(-self.r * self.t_1)
        
        return l
    
    def dou_value(self, n) :
        x_1 = np.polynomial.legendre.leggauss(100)[0]
        w_1 = np.polynomial.legendre.leggauss(100)[1]
        x_2 = np.polynomial.legendre.leggauss(100)[0]
        w_2 = np.polynomial.legendre.leggauss(100)[1]
        l = 0
        for i in range(len(x_1)) :
            t_3 = (x_1[i] + 1) * (self.mu_1 + 10 * self.sigma_1 - self.k) / 2 + self.k
            w_3 = (self.mu_1 + 10 * self.sigma_1 - self.k) / 2
    
            for j in range(len(x_1)) :
                t_4 = (x_2[j] + 1) * (n - self.mu_2 + 10 * self.sigma_2) / 2 + self.mu_2 - 10 * self.sigma_2
                w_4 = (n - self.mu_2 + 10 * self.sigma_2) / 2
                a = w_3 * w_4 * self.p(t_3, t_4) * w_1[i] * w_2[j] * (t_3 - self.k)
                l += a
                
        return l * np.exp(-self.r * self.t_1)
    
    
if __name__ == "__main__" :
    a = Eurocall(10, 12, 0.25, 0.20, 0.04)    
    b = JNormal(323.5, 20, 15, 0.95, 0, 1, 0.5, 370)
    c = JNormal(323.5, 20, 15, 0.8, 0, 1, 0.5, 370)
    d = JNormal(323.5, 20, 15, 0.5, 0, 1, 0.5, 370)
    e = JNormal(323.5, 20, 15, 0.2, 0, 1, 0.5, 370)
    f = JNormal(323.5, 20, 15, 0, 0, 1, 0.5, 370)
    
    
    
    
    
    
    
    
    
    
    
    
    