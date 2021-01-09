#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:32:35 2020

@author: Yuyang Zhang
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class MCStockSimulator :
    '''encapsulates the data and methods required to simulate
       stock returns and values'''
    def __init__(self, s, t, r, sigma, beta, nper_per_year) :
        '''s : the current stock price in dollars
           t : the option maturity time in years
           r : the annualized rate of return on this stock
           sigma : the annualized standard deviation of returns
           nper_per_year : the number of discrete time periods per year'''
        self.s = s
        self.t = t
        self.r = r
        self.sigma = sigma
        self.beta = beta
        self.p = nper_per_year
    
    def __repr__(self) :
        '''returns a string representation for this class'''
        rep = "StockSimulator (s=$%.2f, t=%.2f (years), r=%.2f, sigma=%.2f, beta=%.2f, nper_per_year=%d)"%(
                self.s, self.t, self.r, self.sigma, self.beta, self.p)
        
        return rep
    
    def generate_simulated_stock_values(self) :
        '''generate and return a np.array containing a sequence of simulated stock returns
           over the time period t'''
        zs = np.random.normal(0, 1, int(self.t * self.p))
        dt = 1 / self.p
        si_return = [0 for z in zs]
        si_value = [0 for i in range(len(zs) + 1)]
        si_value[0] = self.s
#        print(len(si_return))
        for i in range(len(si_return)) :
            si_return[i] = self.r * dt + self.sigma * (si_value[i] ** (self.beta - 1)) * zs[i] * math.sqrt(dt)
#            print(si_return[i])
            si_value[i + 1] = si_value[i] * (1 + si_return[i])
        
        return np.array(si_value)
    
    def plot_simulated_stock_values(self, num_trials = 1) :
        '''generate a plot of num_trials series of simulated stock returns
           num_trials is an optional parameter, if it is not supplied, the
           default value of 1 will be used.'''
        data = []
        
        for i in range(num_trials) :
            data += [self.generate_simulated_stock_values()]
        data = np.array(data)
        
#        print(data)
        ts = np.arange(0, self.t + 1 /self.p, 1 / self.p)
        
#        print(ts)
        plt.plot(ts, data.T)
        plt.xlabel("years")
        plt.ylabel("$ value")
        plt.title("%d simulated trials"% num_trials)
        

class MCEuroCallOption(MCStockSimulator) :
    def __init__(self, s, x, t, r, sigma, beta, nper_per_year, number_trials) :
        
        super().__init__(s, t, r, sigma, beta, nper_per_year)
#        print(self.p)
        self.x = x
        self.nt = number_trials

    def __repr__(self) :
        '''a new version of the string representation for this subclass'''
        rep = "MCEuroCallOption, s=%.2f, x=%.2f, t=%.2f, r=%.2f, sigma=%.2f, beta=%.2f nper_per_year=%d, num_trials=%d"%(
                self.s, self.x, self.t, self.r, self.sigma, self.beta, self.p, self.nt)
        return rep
    
    def value(self) :
        trials = []
        stock = []
        for i in range(self.nt) :
            s_t = self.generate_simulated_stock_values()[-1]
            stock += [s_t]
            c = max(s_t - self.x, 0) * math.exp(-self.r * self.t)
            # print(c)
            trials += [c]
        stock = np.array(stock)
        trials = np.array(trials)
        plt.scatter(stock, trials)
        # plt.hist(trials)
            
        
        
        return np.mean(trials)

class Portfolio(MCStockSimulator):
    def __init__(self, s, x, t, r, sigma, beta, nper_per_year, unit, delta, n_trials) :
        super().__init__(s, t, r, sigma, beta, nper_per_year)
        self.unit = unit
        self.delta = delta
        self.ntrials = n_trials
        self.x = x
    
    def payoff(self) :
        trial = []
        for i in range(self.ntrials) :
            s_t = self.generate_simulated_stock_values()[-1] 
            # print(s_t)
            s_t1 = (s_t - self.s) * self.unit * self.delta
            c_t = max(s_t - self.x, 0) * self.unit
            # print(c_t)
            payoff = c_t - s_t1
            trial += [payoff]
            
        
        return np.mean(trial) * math.exp(-self.r * self.t)
        
    def plot(self) :
        trial = []
        stock = []
        for i in range(self.ntrials) :
            s_t = self.generate_simulated_stock_values()[-1]
            # print(type(s_t))
            # print(s_t)
            stock += [s_t]
            s_t1 = (s_t- self.s) * self.unit * self.delta
            c_t = max(s_t - self.x, 0) * self.unit
            # print(c_t)
            payoff = c_t - s_t1
            trial += [payoff]
        stock = np.array(stock)
         # print(stock)
        payoff = np.array(trial)
        # print(payoff)
        plt.scatter(stock, payoff)
        # plt.hist(payoff)
         
         
    
        
    

if __name__ == "__main__" :
    call = MCEuroCallOption(100, 100, 1, 0, 0.25, 1, 100, 100000)
    sim = MCStockSimulator(100, 1, 0, 0.25, 0.5, 250)
    portfolio = Portfolio(100, 100, 1, 0, 0.25, 1, 100, 1, 0.55, 100000)
    portfolio_1 = Portfolio(100, 100, 1, 0, 0.25, 0.5, 100, 1, 0.55, 100000)
    portfolio_2 = Portfolio(100, 100, 1, 0, 0.4, 1, 100, 1, 0.579, 100000)
    call_1 = MCEuroCallOption(100, 100, 1, 0, 0.4, 1, 100, 100000)
#    print(sim)
    # sim.plot_simulated_stock_values(5)
    
        




















