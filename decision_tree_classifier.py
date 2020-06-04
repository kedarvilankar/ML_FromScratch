# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:58:29 2020

@author: Kedarpv
Decision tree from scratch
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from scipy import stats

def gini_impurity(y):
    """
    Computes Gini impurity in y

    Parameters
    ----------
    y : Categoriacal
        Target variable.

    Returns
    -------
    impurity : float
        Gini impurity in y.

    """
    classes = set(y)
    noClasses = len(classes)
    probs_sq = 0
    for cl in classes:
        probs_sq += (sum(y==cl)/len(y))**2
        
    impurity = 1-probs_sq
    return impurity
        
def get_impurity_reduction(x,y,gini_p,decision_levels):
    """
    Computes impurity reduction in child nodes at different dicision levels of x (one of the features in X).

    Parameters
    ----------
    x : float
        x feature for which impurity reduction to be computed.
    y : Categoriacal
        Target variable.
    gini_p : float
        Parent nodes Gini impurity.
    decision_levels : int
        Number of decision levels.

    Returns
    -------
    impurity_reduction : float
        Array containing impurity reduction at various decision levels of x.

    """
    len_y = len(y)
    decisions = np.linspace(x.min(),x.max(),decision_levels)
    impurity_reduction = []
    for d in decisions:
        y_left = y[x<=d]
        y_right = y[x>d]
        reduction = gini_p - ((len(y_left)/len_y) *gini_impurity(y_left)) - ((len(y_right)/len_y) *gini_impurity(y_right))
        impurity_reduction.append(reduction)
    
    return impurity_reduction

def get_impurity_reduction_allFeatures(X,y,gini_p,decision_levels):
    """
    Computes impurity reduction in child nodes at different dicision levels of of all features.

    Parameters
    ----------
    X : TYPE
        X training set.
    y : TYPE
        Target variable.
    gini_p : float
        Parent node's Gini impurity.
    decision_levels : int
        Number of decision levels.

    Returns
    -------
    impurity_reduction : float
        Impurity reduction for each feature at different decision levels.

    """
    no_features = X.shape[1]
    impurity_reduction = []
    for i in range(no_features):
        reduction_i = get_impurity_reduction(X[:,i],y,gini_p,decision_levels)
        impurity_reduction.append(reduction_i)
    
    return impurity_reduction

def make_decision(X,y,gini_p,min_samples_split,decision_list,curent_node,decision_levels=5):
    """
    Make a decision to split the node.

    Parameters
    ----------
    X : TYPE
        X training set.
    y : TYPE
        Target variable.
    gini_p : float
        Parent node's Gini impurity.
    min_samples_split : int
        Minimum number of samples in a node to split further.
    decision_list : List
        List of decision at each node.
    curent_node : int
        Node number in binary tree.
    decision_levels : int, optional
       The number of decision levels to evaluate for each feature. The default is 5.

    Returns
    -------
    None.

    """
    impurity_reduction = np.array(get_impurity_reduction_allFeatures(X,y,gini_p,decision_levels))
    #find best decisions feature number and level number
    (f,l) = np.unravel_index(impurity_reduction.argmax(), impurity_reduction.shape)
    #print(impurity_reduction)
    
    #split tree with best decision, but check first if max depth is acheived
    left_child_node =  curent_node*2 + 1 
    right_child_node =  curent_node*2 + 2 
    
    if left_child_node<len(decision_list): #check if max_depth reached
    
        len_y = len(y)
        all_decisions = np.linspace(X[:,f].min(),X[:,f].max(),decision_levels)
        best_decsion = all_decisions[l]
        
        y_left = y[X[:,f]<=best_decsion]
        y_right = y[X[:,f]>best_decsion]
        
        decision_list[curent_node] = (f,best_decsion,len_y)
        
        # check if children are leaf nodes or split further
        if len(y_left)>=min_samples_split and gini_impurity(y_left)!=0:
            make_decision(X[X[:,f]<=best_decsion,:],y_left,gini_p,min_samples_split,decision_list,left_child_node)
        elif gini_impurity(y_left)==0: # leaf node with pure class
            decision_list[left_child_node] = (np.nan,np.nan,len(y_left),np.unique(y_left)[0])
        else: # reached min_samples_split
            if len(y_left): # check if there are no sample
                decision_list[left_child_node] = (np.nan,np.nan,len(y_left),stats.mode(y_left)[0][0])
            
        # check if children are leaf nodes or split further
        if len(y_right)>=min_samples_split and gini_impurity(y_right)!=0:
            if right_child_node<len(decision_list): # max_depth reached
                make_decision(X[X[:,f]>best_decsion,:],y_right,gini_p,min_samples_split,decision_list,right_child_node)
        elif gini_impurity(y_right)==0: # leaf node with pure class
            decision_list[right_child_node] = (np.nan,np.nan,len(y_right),np.unique(y_right)[0])
        else: # reached min_samples_split
            if len(y_right): # check if there are no sample
                decision_list[right_child_node] = (np.nan,np.nan,len(y_right),stats.mode(y_right)[0][0])
            
    else: # force current node leaf
        decision_list[curent_node] = (np.nan,np.nan,len(y),stats.mode(y)[0][0])
                

def fitDT(X,y, min_samples_split=2,max_depth = 10):
    """
    Fit decision tree classfier on X and y data.

    Parameters
    ----------
    X : TYPE
        X training set.
    y : TYPE
    min_samples_split : int
        Minimum number of samples in a node to split further. The default is 2.
    max_depth : int, optional
        Maximum depth the decision binary can have. The default is 10.

    Returns
    -------
    decision_list : List
        List of decision at each node.

    """
    no_nodes = 2**(max_depth) - 1
    decision_list = [[]]*no_nodes
    
    curent_node = 0
    gini_p = gini_impurity(y)
    print(gini_p)
    
    if len(y)>=min_samples_split and gini_p!=0:
        make_decision(X,y,gini_p,min_samples_split,decision_list,curent_node)
    
    return decision_list

iris = load_iris()
X = iris.data[:,2:]
y = iris.target


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# clf = DecisionTreeClassifier()
# clf.fit(X_train,y_train)

# y_pred = clf.predict(X_test)
# print('SKLEARN RESULTS\n',classification_report(y_test,y_pred))
decision_list = fitDT(X_train,y_train)