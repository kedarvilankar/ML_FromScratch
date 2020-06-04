# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:30:57 2020

@author: Kedarpv

Linear Regression from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def make_data(n=200):
    """
    Generate data with a linear trend   

    Parameters
    ----------
    n : int, optional
        The number of samples to be generated. The default is 200.

    Returns
    -------
    x : float
        Independent variable data
    y : float
        Dependent variable/target variable
    m : float
        Slope of the linear trend
    c : float
        y-intercept

    """
    m = np.random.rand()*10
    c = np.random.rand()*10
    
    x = np.random.random_sample(n) 
    noise = np.random.randn(n)
    y = m*x + c + 1.5*noise
    
    return (x,y,m,c)

def predict(x,m,c):
    """
    Predict y with given x,m, and c.

    Parameters
    ----------
    x : float
        Independent variable data
    m : float
        Slope of the linear trend
    c : float
        y-intercept

    Returns
    -------
    float
        predicted target values(y^).

    """
    return x*m+c

def grad_m(error,x_train):
    """
    Compute gradient with respect to m.

    Parameters
    ----------
    error : float
        Error in prediction.
    x_train : float
        Training data

    Returns
    -------
    float
        Gradient with respet to m.

    """
    return np.mean(error*-x_train)

def grad_c(error):
    """
    Compute gradient with respect to c.

    Parameters
    ----------
    error : float
        Error in prediction.

    Returns
    -------
    float
        Gradient with respet to c.

    """
    return np.mean(-error)

def fit_model(x_train, y_train, lr=0.001, n_iter = 100):
    """
    Fit a linear model on x_train and y_train

    Parameters
    ----------
    x_train : float
        Training data
    y_train : float
        target variable from training set
    lr : float, optional
        Learning rate. The default is 0.001.
    n_iter : int, optional
        Number of iteration to optimize the model. The default is 100.

    Returns
    -------
    m : TYPE
        Predicted slope.
    c : TYPE
        Predicted y-intercept.
    cost : TYPE
        COst on training set.

    """
    m = np.random.rand()*10
    c = np.random.rand()*10
    
    cost = np.zeros((n_iter,1))
    for i in range(n_iter):
        y_pred = predict(x_train,m,c)
        error = y_train-y_pred
        cost[i] = np.mean(error**2)
        del_m = grad_m(error,x_train)
        del_c = grad_c(error)
        
        #update m and C
        m = m-lr*del_m
        c = c-lr*del_c
        
    return (m,c,cost)

#Generate data    
x,y,act_m,act_c = make_data()
#plt.scatter(x,y)

# Split data into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Fit a linear model
m,c,cost = fit_model(x_train,y_train,lr=0.1, n_iter = 100)

#test error
y_pred = predict(x_test,m,c)

print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))
x_plot = np.linspace(x_test.min(),x_test.max(),100)
y_plot_pred = predict(x_plot,m,c)
y_plot_act = predict(x_plot,act_m,act_c)

plt.scatter(x_test,y_test)
plt.plot(x_plot,y_plot_act, 'g')
plt.plot(x_plot,y_plot_pred, 'r')
   
