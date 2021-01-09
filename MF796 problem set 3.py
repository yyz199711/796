#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:30:17 2020

@author: Yuyang Zhang
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import scipy.stats as si
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splrep, splev
import time



class FFTH :
    
    def __init__(self, s, k, t, r, sigma, nu, kappa, rho, theta, q) :
        
        self.s = np.log(s)
        self.k = np.log(k)
        self.t = t
        self.r = r
        self.sigma = sigma
        self.nu = nu
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        # self.alpha = alpha
        self.q = q
    
    
    def lamba(self, u) :
        
        return np.sqrt(self.sigma ** 2 * (u ** 2 + complex(0,1) * u) + (self.kappa - complex(0,1) * self.rho * self.sigma * u) ** 2)
    
    
    def omega(self, u) :
        
        i = complex(0, 1)
        
        a = self.kappa * self.theta * self.t * (self.kappa - i * self.rho * self.sigma * u) / self.sigma ** 2
        
        b = np.cosh(self.lamba(u) * self.t / 2) + (self.kappa - i * self.rho * self.sigma * u) / self.lamba(u) * np.sinh(self.lamba(u) * self.t / 2)
        
        d = 2 * self.kappa * self.theta / self.sigma ** 2
        
        e = b ** d
        
        omega = np.exp(i * u * self.s + i * u * (self.r - self.q) * self.t + a) / e
        
        return omega
    
    def Phi(self, u) :
        
        i = complex(0, 1)
        
        f = np.cosh(self.lamba(u) * self.t / 2) / np.sinh(self.lamba(u) * self.t / 2)
        
        c = -(u ** 2 + i * u) * self.nu / (self.lamba(u) * f + self.kappa - i * self.rho * self.sigma * u)
    
        return self.omega(u) * np.exp(c)
    
    
    def callphi(self, v, alpha) :
        
        i = complex(0, 1)
        
        x = v - (alpha + 1) * i
        
        coef = np.exp(-self.r * self.t) / ((alpha + i * v) * (alpha + i * v + 1))
        
        return coef * self.Phi(x)
    
    def CallPrice(self, alpha, n, b) :
        coef = np.exp(-alpha * self.k) / np.pi
        
        dv = b / n
        
        i = complex(0, 1)
        
        a = 0
        
        for j in range(1, n + 1) : 
            
            v_1 = (j - 1) * dv
            
            v_2 = j * dv
            
            a += coef * 0.5 * dv * (np.exp(-i * v_1 * self.k) * self.callphi(v_1, alpha) + np.exp(-i * v_2 * self.k) * self.callphi(v_2, alpha))
            
            
            
        return a.real
    
    
    def plot_alpha1(self, n, b) :
        
        ran = np.linspace(2, 40, 1000)
        
        # print(ran)
        
        price = np.array([self.CallPrice(j, n, b) for j in ran])
        
        # print(price)
        
        plt.plot(ran, price)
    
    def plot_alpha2(self, n, b) :
        
        ran = np.linspace(0.25, 2, 100)
        
        # print(ran)
        
        price = np.array([self.CallPrice(j, n, b) for j in ran])
        
        # print(price)
        
        plt.plot(ran, price)
    
        
    
    def delta(self, j) :
        
        if j == 1 :
            return 1
        return 0
    
    def pa_x(self, n, b, j, alpha) :
        
        dv = b / n
        v_j = (j - 1) * dv
        
        dk = 2 * np.pi / (n * dv)
        
        i = complex(0, 1)
        
        coef = (2 - self.delta(j)) * dv * np.exp(-i * (self.s - dk * n / 2) * v_j) / 2
        
        return coef * self.callphi(v_j, alpha)
    
    def pa_y(self, n, b, alpha) :
        
        x = np.array([self.pa_x(n, b, j, alpha) for j in range(1, n + 1)])
        
        y = fft(x)
        
        return y.real
    
    # def Call(self, n, b, j, alpha) :
        
    #     dv = b / n
        
    #     dk = 2 * np.pi / (n * dv)
        
    #     call = np.exp(-alpha * (self.s - dk * (n / 2 - (j - 1)))) * self.pa_y(n, b, alpha)[j - 1].real / np.pi
        
    #     # print(call)
    #     return call
    
    def Call(self, n, b, alpha) :
        dv = b / n
        
        dk = 2 * np.pi / (n * dv)
        
        coef = np.array([np.exp(-alpha * (self.s - dk * (n / 2 - (j - 1)))) / np.pi for j in range(1, n + 1)])
        
        k_j = np.array([self.s - dk * (n / 2 - (j - 1)) for j in range(1, n + 1)])
        
        call = coef * self.pa_y(n, b, alpha)
        
        
        tck = splrep(k_j, call)
        
        p = splev(self.k, tck)
        
        # p = p[0]
        # print(call)
        return float(p)

        
    
    
    def plot_nb(self) :
        
        fig = plt.figure()
        ax = Axes3D(fig)
        n = np.array([i for i in range(5, 15)])
        b = np.array([i for i in range(60, 70)])
        p = []
        for m in n :
            d = []
            for l in b :
                a = self.Call(2 ** m, 10 * l, 1)
                # print(a)
                
                d += [a]
            
            p += [d]
        p = np.array(p)
                
        n, b = np.meshgrid(n, b)
        
        ax.plot_surface(n, b, p.T, rstride=1, cstride=1, cmap='rainbow')
        ax.set_zlabel('European Call option price')  
        ax.set_ylabel('upper bound divided by 10')
        ax.set_xlabel('n')
    
    
    def run_time(self) :
        fig = plt.figure()
        ax = Axes3D(fig)

        n = np.array([i for i in range(5, 15)])
        b = np.array([i for i in range(60, 70)])
        
        p = []
        for m in n :
            d = []
            for l in b :
                bt = time.time()
                
                price = self.Call(2 ** m, 10 * l, 1)
                mse = (price - 21.27) ** 2
                
                et = time.time()
                
                a = et - bt
                
                a = 1 / (a * mse)
                
                
                # print(a)
                
                d += [a]
            
            p += [d]
        p = np.array(p)
                
        n, b = np.meshgrid(n, b)
        
        ax.plot_surface(n, b, p.T, rstride=1, cstride=1, cmap='rainbow')
        ax.set_zlabel('Running time with respect to different N and B')  
        ax.set_ylabel('upper bound divided by 10')
        ax.set_xlabel('n')
        
    # def plot_b(self) :
    #     b = np.array([i for i in range(1, 20)])
        
    #     c = []
        
    #     for i in b :
    #         bt = time.time()
    #         price = self.Call(2 ** 8, i, 1)
            
    #         et = time.time()
            
    #         a = et - bt
            
    #         c += [a]
        
    #     c = np.array(c)
        
    #     plt.plot(b, c)
    #     plt.xlabel("Upper bound divided by 5")
    #     plt.ylabel("Runing time with respect to B")
        
            

        
        
        

        
        
    
    # def error(self) :
    #     a = []
    #     p = self.CallPrice(1, 100, 30)
    #     for n in range(5, 16) :
    #         c = []
    #         for b in range(1, 21):
    #             err = abs(self.Call(2 ** n, 50 * b, 2 ** n // 2 + 1, 1) - p)
    #             print(err)
    #             d = [n, b]
    #             c += [err, d]
    #         a += [c]
    #     return min(a)
        
                
                
    # def plot_n(self) :
        
    #     call = 
        
    
    # def plot_b2(self)
        
    
    # def Strike(self, n, b, alpha) :
        
    #     dv = b / n
        
    #     dk = 2 * np.pi / (n * dv)
        
    #     coef = np.array([np.exp(-alpha * (self.s - dk * (n / 2 - (j - 1)))) for j in range(1, n + 1)])
        
    #     strike = coef * self.pa_y(n, b, alpha).real / np.pi
        
    #     return strike
        
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

class FFT2 :
    
    def __init__(self, s, r, q, alpha) :
        
        self.s = s
        self.r = r
        self.q = q
        self.alpha = alpha
        
    def im_vol_1(self, k, t, sigma, nu, kappa, rho, theta) :
        upper_bound = 1
        lower_bound = 0.0001
        guess = 0.5
        a = Eurocall(self.s, k, t, guess, self.r - self.q).BScall()
        b = FFTH(self.s, k, t, self.r, sigma, nu, kappa, rho, theta, self.q).CallPrice(self.alpha, 100, 30)
        # print(b)
        while abs(a - b) > 0.001 :
            if a > b :
                upper_bound = guess
                guess = (upper_bound + lower_bound) / 2
            
            if a < b :
                lower_bound = guess
                guess = (lower_bound + upper_bound) / 2
                
            
            
            a = Eurocall(self.s, k, t, guess, self.r - self.q).BScall()
            # print(guess, a)
        
        return guess
            
        
    
    def plot_k(self, t, sigma, nu, kappa, rho, theta) :
        
        k = np.arange(90, 200, 5)
        a = []
        for i in k :
            c = self.im_vol_1(i, t, sigma, nu, kappa, rho, theta)
            # print(c)
            
            a += [c]
        
        a = np.array(a)
        
        
        plt.plot(k, a)
        plt.xlabel('K')
        plt.ylabel('Implied Volatility')
        
    
    def im_vol_2(self, k, t, sigma, nu, kappa, rho, theta) :
        upper_bound = 1
        lower_bound = 0.0001
        guess = 0.5
        a = Eurocall(self.s, k, t, guess, self.r - self.q).BScall()
        b = FFTH(self.s, k, t, self.r, sigma, nu, kappa, rho, theta, self.q).Call(512, 600, 1)
        # print(b)
        while abs(a - b) > 0.001 :
            if a > b :
                upper_bound = guess
                guess = (upper_bound + lower_bound) / 2
            
            if a < b :
                lower_bound = guess
                guess = (lower_bound + upper_bound) / 2
                
            
            
            a = Eurocall(self.s, k, t, guess, self.r - self.q).BScall()
            # print(guess, a)
        
        return guess
    
    def plot_price(self, k, sigma, nu, kappa, rho, theta) :
        t = np.arange(0.1, 2, 0.1)
        a = []
        for i in t :
            c = FFTH(self.s, k, i, self.r, sigma, nu, kappa, rho, theta, self.q).Call(512, 600, 1.5)
            # print(c)
            a += [c]
        
        a = np.array(a)
        
            
        
        
        plt.plot(t, a)
        plt.xlabel("Time")
        plt.ylabel("FFT price")
    
    def plot_price1(self, t, sigma, nu, kappa, rho, theta) :
        k = np.arange(90, 200, 5)
        a = []
        for i in k :
            c = FFTH(self.s, i, t, self.r, sigma, nu, kappa, rho, theta, self.q).Call(512, 600, 1.5)
            # print(c)
            a += [c]
        
        a = np.array(a)
        
            
        
        
        plt.plot(k, a)
        plt.xlabel("Strike K")
        plt.ylabel("FFT price")

        
    
    def plot_t(self, k, sigma, nu, kappa, rho, theta) :
        
        t = np.arange(0.1, 4, 0.1)
        a = []
        for i in t :
            c = self.im_vol_2(150, i, 0.4, 0.09, 0.5, 0.25, 0.12)
            # print(c)
            a += [c]
        
        a = np.array(a)
        
            
        
        
        plt.plot(t, a)
        plt.xlabel("Time")
        plt.ylabel("Implied Volatility")
        
        
    def plot_sigma1(self, nu, kappa, rho, theta) :
        
        sigma = [0.3, 0.4, 0.5, 0.6]
        k = np.arange(90, 200, 5)
        e = []
        for i in sigma :
            d = []
            
            for m in k :
                
                f = self.im_vol_1(m, 0.25, i, nu, kappa, rho, theta)
                # print(f)
                
                d += [f]
            
            e += [d]
        
        e = np.array(e)
        plt.plot(k, e.T)
        plt.xlabel("K")
        plt.ylabel("Implied Volatility")
        plt.legend(["sigma = 0.3", "sigma = 0.4", "sigma = 0.5", "sigma = 0.6"])
      
    
        
    def plot_sigma2(self, nu, kappa, rho, theta):
        e = []
        sigma = [0.2, 0.4, 0.6, 0.8]
        t = np.arange(0.1, 2, 0.1)
        for i in sigma :
            d = []
            
            for m in t :
                
                f = self.im_vol_2(150, m, i, nu, kappa, rho, theta)
                # print(f)
                
                d += [f]
            
            e += [d]
        
        e = np.array(e)
        plt.plot(t, e.T)
        plt.xlabel("Time")
        plt.ylabel("Implied Volatility")
        plt.legend(["sigma = 0.3", "sigma = 0.4", "sigma = 0.5", "sigma = 0.6"])
        
        
    
    def plot_nu1(self, sigma, kappa, rho, theta) :
        
        nu = [0.08, 0.09, 0.1, 0.11]
        k = np.arange(95, 200, 5)
        
        a = []
        for i in nu :
            b = []
            for j in k :
            
                c = self.im_vol_1(j, 0.25, sigma, i, kappa, rho, theta)
                # print(c)
            # print(c)
                b += [c]
            
            a += [b]
        
        a = np.array(a)
        
        plt.plot(k, a.T)
        plt.xlabel("K")
        plt.ylabel("Implied Volatility")
        plt.legend(["nu = 0.08", "nu = 0.09", "nu = 0.10", "nu = 0.11"])

    def plot_nu2(self, sigma, kappa, rho, theta) :
        t = np.arange(0.1, 2, 0.05)
        nu = [0.08, 0.09, 0.10, 0.11]
        e = []
        for i in nu :
            d = []
            
            for m in t :
                
                f = self.im_vol_2(150, m, sigma, i, kappa, rho, theta)
                
                d += [f]
            
            e += [d]
        
        e = np.array(e)
        plt.plot(t, e.T)
        plt.xlabel("Time")
        plt.ylabel("Implied Volatility")
        plt.legend(["nu = 0.08", "nu = 0.09", "nu = 0.10", "nu = 0.11"])
        
        
    def plot_kappa1(self, sigma, nu, rho, theta) :
        
        kappa = [0.3, 0.5, 0.7, 0.9]
        k = np.arange(95, 200, 5)
        
        a = []
        for i in kappa :
            b = []
            for j in k :
            
                c = self.im_vol_1(j, 0.25, sigma, nu, i, rho, theta)
                # print(c)
            # print(c)
                b += [c]
            
            a += [b]
        
        a = np.array(a)
        
        plt.plot(k, a.T)
        plt.xlabel("K")
        plt.ylabel("Implied Volatility")
        plt.legend(["Kappa = 0.3", "Kappa = 0.5", "Kappa = 0.7", "Kappa = 0.9"])
        
        # a = np.array([self.im_vol_2(k, t, sigma, nu, i, rho, theta) for i in kappa])
        
        # plt.plot(kappa, a)
        # plt.xlabel("Kappa")
        # plt.ylabel("Implied Volatility")
        
    def plot_kappa2(self, sigma, nu, rho, theta) :
        t = np.arange(0.1, 2, 0.05)
        kappa = [0.3, 0.5, 0.7, 0.9]
        e = []
        for i in kappa :
            d = []
            
            for m in t :
                
                f = self.im_vol_2(150, m, sigma, nu, i, rho, theta)
                
                d += [f]
            
            e += [d]
        
        e = np.array(e)
        plt.plot(t, e.T)
        plt.xlabel("Time")
        plt.ylabel("Implied Volatility")
        plt.legend(["Kappa = 0.3", "Kappa = 0.5", "Kappa = 0.7", "Kappa = 0.9"])

        

        
    def plot_rho1(self, sigma, nu, kappa, theta) :
        
        rho = [-0.5, -0.25, 0, 0.25, 0.5]
        
        k = np.arange(95, 200, 5)
        
        a = []
        for i in rho :
            b = []
            for j in k :
            
                c = self.im_vol_1(j, 0.25, sigma, nu, kappa, i, theta)
                # print(c)
            # print(c)
                b += [c]
            
            a += [b]
        
        a = np.array(a)
        
        plt.plot(k, a.T)
        plt.xlabel("K")
        plt.ylabel("Implied Volatility")
        plt.legend(["Rho = -0.5", "Rho = -0.25", "Rho = 0", "Rho = 0.25", "Rho = 0.5"])
    
    def plot_rho2(self, sigma, nu, kappa, theta) :
        t = np.arange(0.1, 2, 0.05)
        rho = [-0.5, -0.25, 0, 0.25,0.5]
        e = []
        for i in rho :
            d = []
            
            for m in t :
                
                f = self.im_vol_2(150, m, sigma, nu, kappa, i, theta)
                
                d += [f]
            
            e += [d]
        
        e = np.array(e)
        plt.plot(t, e.T)
        plt.xlabel("Time")
        plt.ylabel("Implied Volatility")
        plt.legend(["Rho = -0.5", "Rho = -0.25", "Rho = 0", "Rho = 0.25", "Rho = 0.5"])

        
    def plot_theta1(self, sigma, nu, kappa, rho) :
        
        theta = [0.08, 0.10, 0.12, 0.14, 0.16]
        
        k = np.arange(95, 200, 5)
        
        a = []
        for i in theta :
            b = []
            for j in k :
            
                c = self.im_vol_1(j, 0.25, sigma, nu, kappa, rho, i)
                # print(c)
            # print(c)
                b += [c]
            
            a += [b]
        
        a = np.array(a)
        
        plt.plot(k, a.T)
        plt.xlabel("K")
        plt.ylabel("Implied Volatility")
        plt.legend(["Theta = 0.08", "Theta = 0.10", "Theta = 0.12", "Theta = 0.14", "Theta = 0.16"])

    def plot_theta2(self, sigma, nu, kappa, rho) :
        theta = [0.08, 0.10, 0.12, 0.14, 0.16]
        t = np.arange(0.1, 2, 0.05)
        e = []
        for i in theta :
            d = []
            
            for m in t :
                
                f = self.im_vol_2(150, m, sigma, nu, kappa, rho, i)
                
                d += [f]
            
            e += [d]
        
        e = np.array(e)
        plt.plot(t, e.T)
        plt.xlabel("Time")
        plt.ylabel("Implied Volatility")
        plt.legend(["Theta = 0.08", "Theta = 0.10", "Theta = 0.12", "Theta = 0.14", "Theta = 0.16"])


        
    
        
    
            
        
    
            
            
        
    
        
        

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    















if __name__ == "__main__" :
    a = FFTH(250, 250, 0.5, 0.02, 0.2, 0.08, 0.7, -0.4, 0.1, 0)
    b = FFTH(150, 150, 1, 0.025, 0.4, 0.09, 0.5, 0.25, 0.12, 0)
    c = FFT2(150, 0.025, 0, 1.5)
    d = Eurocall(150, 100, 0.25, 0.3, 0.025)























        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
