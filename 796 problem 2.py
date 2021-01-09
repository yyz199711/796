#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:21:24 2020

@author: Yuyang Zhang
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
        self.end = np.exp(self.u + 6 * self.sigma * np.sqrt(self.t))
    
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
        density = 1 / (x * self.sigma * np.sqrt(2 * np.pi * self.t)) * np.exp(-np.power(np.log(x) - self.u, 2) / (2 * np.power(self.sigma, 2) * self.t))
        # print(density)
        return density
        
    
    def lrm(self, n) :
        w = (self.end - self.k) / n
        l = 0
        for i in range(n) :
            x = self.k + i * w
            l += self.den(x) * w * (x - self.k)
        
        return np.exp(-self.r * self.t) * l
    
    
    
    def lrmerr(self, n) :
        
        return self.lrm(n) - self.BScall()
    
    def mrm(self, n) :
        w = (self.end - self.k) / n
        l = 0
        for i in range(n) :
            x = self.k + i * w + 0.5 * w
            l += self.den(x) * (x - self.k) * w
        
        return np.exp(-self.r * self.t) * l
        
    
    def mrmerr(self, n) :
        
        return self.mrm(n) - self.BScall()
    
    def Gauss(self, n) :
        w = np.polynomial.legendre.leggauss(n)[1]
        x = np.polynomial.legendre.leggauss(n)[0] 
        l = 0
        for i in range(len(x)) :
            t = (x[i] + 1) * (self.end - self.k) / 2 + self.k
            w_1 = (self.end - self.k) / 2
            l += self.den(t) * (t - self.k) * w[i] * w_1
            

        return np.exp(-self.r * self.t) * l
    
    
    def Gausserr(self, n) :
        
        return self.Gauss(n) - self.BScall()
    
    
    def plotlrm(self) :
        y = np.array([abs(self.lrmerr(i)) for i in range(5, 101)])
        x_1 = np.array([1 / i * 5 * abs(self.lrmerr(5)) for i in range(5, 101)])
        x_2 = np.array([(5 / i) ** 2 * abs(self.lrmerr(5)) for i in range(5, 101)])
        x_3 = np.array([(5 / i) ** 3 * abs(self.lrmerr(5)) for i in range(5, 101)])
        plt.plot(y)
        plt.plot(x_1)
        plt.plot(x_2)
        plt.plot(x_3)
        plt.legend(['Left-Riemann', '1/N', '1/N^2', '1/N^3'])
        
    def plotmrm(self) :
        y = np.array([abs(self.mrmerr(i)) for i in range(5, 101)])
        # x_1 = np.array([1/ i for i in range(5, 101)])
        x_2 = np.array([(5 / i) ** 2 * abs(self.mrmerr(5)) for i in range(5, 101)])
        x_3 = np.array([(5 / i) ** 3 * abs(self.mrmerr(5)) for i in range(5, 101)])
        plt.plot(y)
        # plt.plot(x_1)
        plt.plot(x_2)
        plt.plot(x_3)
        plt.legend(['Mid-rule', '1/N^2', '1/N^3'])
        
    def plotGauss(self) :
        y = np.array([abs(self.Gausserr(i)) for i in range(5, 101)])
        # x_1 = np.array([1/ i for i in range(5, 101)])
        x_2 = np.array([i ** (-2 * i) * 5 ** 10 * abs(self.Gausserr(5))  for i in range(5, 101)])
        # x_3 = np.array([1 / i ** 3 for i in range(5, 101)])
        plt.plot(y)
        # plt.plot(x_1)
        plt.plot(x_2)
        # plt.plot(x_3)
        plt.legend(['Gauss nodes', 'N^{-2N}'])   
            
    # def plot(self) :
        
    #     x = np.array([i for i in range(1, 101)])
    #     y_1 = np.array([1000 *self.lrmerr(i) for i in range(1, 101)])
    #     y_2 = np.array([1000 * self.mrmerr(i) for i in range(1, 101)])
    #     y_3 = np.array([100000 * self.Gausserr(i) for i in range(1, 101)])
        
    #     plt.scatter(x, y_1)
    #     plt.scatter(x, y_2)
    #     plt.scatter(x, y_3)
            
        
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
        
        p = 1 / (2 * np.pi * self.sigma_1 * self.sigma_2 * np.sqrt(1 - self.rho ** 2)) * np.exp(- z / 2 * (1 - np.power(self.rho, 2)))
                                                                                                
        return p                                                                                        
    
    def sin_value(self) :
        x = np.polynomial.hermite.hermgauss(100)[0]
        w = np.polynomial.hermite.hermgauss(100)[1]
        l = 0
        for i in range(len(x)) :
            f = self.p_1(x[i]) * np.exp(x[i] ** 2) * w[i]
            l += f
        
        l *= np.exp(-self.r * self.t_1)
        
        return l
    
    def dou_value(self, n) :
        w_1 = np.polynomial.laguerre.laggauss(100)[1]
        t = np.polynomial.laguerre.laggauss(100)[0] 
        w_2 = np.polynomial.hermite.hermgauss(100)[1]
        x = np.polynomial.hermite.hermgauss(100)[0]
        l = 0
        for i in range(len(x)) :
    
            for j in range(len(x)) :
                a = w_1[i] * w_2[j] * max(x[j] - self.k, 0) * self.p(x[j], n - t[i]) * np.exp(t[i]) * np.exp(x[j] ** 2)
                l += a
                
        return l * np.exp(-self.r * self.t_1)
    
                
            
        
        
        
        
        
        
        
            
        
        
        
        
        
        


    
    
    
if __name__ == "__main__" :
    a = Eurocall(10, 12, 0.25, 0.20, 0.04)
    b = JNormal(450, 20, 15, 0.95, 0, 1, 0.5, 370)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    