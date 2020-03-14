from classification_dataset import create_data_with_classes as cdc
from cross_validation import nested_cv

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os

SEED = 30

def _load_npy(file):
    '''
    Loading npy file, in the current working directory, with the given name
    Need src as working directory

    Input:
    :param file: The name of the npy file to load
    '''
    path = os.path.join(os.path.join(os.getcwd(), 'william-logistic/npys'), file+'.npy')
    return np.load(path)

def _log(path, payload):
    '''
    Logging function

    Input:
    :param path: Path to the log file
    :param log_func: A function returning a string to be logged
    '''
    with open(path, 'a+') as f:
        f.write(payload)


def base_line(X_train, y_train):
    '''
    Baseline model always predicting for the class which occurs the most in the training data

    Input:
    :param params: Should always be X_train and y_train to match the input for other models

    Output: A baseline model predicting for the most occuring classs in the training data
    '''
    #X_train only with to have similar input for base and other models
    vals, counts = np.unique(y_train, return_counts=True)
    most_common = vals[np.argmax(counts)]

    return lambda x: most_common


def logistic_reg(lamb):
    '''
    Returns a logistic regression function(model) which is trained on the provided X_train and y_train data.
    The model function returned can be used to predict on a new X dataset.

    Input:
    :param lamb: The lambda regularization constant to use for the logistic regression model
    :param params: Should always be X_train and y_train with training data and target data for the model

    Output: A logistic regression function(model)
    '''
    def logi(X_train, y_train):
        # Fit regularized logistic regression model to training data to predict 
        # the risk category
        mdl = LogisticRegression(penalty='l2', C=lamb)
        mdl.fit(X_train, y_train)
        return lambda x: mdl.predict(x)
    
    return logi

def decision_tree(lamb):
    '''
    Returns a decision tree classifier function(model) which is trained on the provided X_train and y_train data.
    The model function returned can be used to predict on a new X dataset.

    Input:
    :param lamb: The lambda regularization constant to use for the logistic regression model
    :param params: Should always be X_train and y_train with training data and target data for the model

    Output: A decision tree classifier function(model)
    '''
    def tree(X_train, y_train):
        # Fit regularized logistic regression model to training data to predict 
        # the risk category
        mdl = DecisionTreeClassifier(min_impurity_decrease=lamb)
        mdl.fit(X_train, y_train)
        return lambda x: mdl.predict(x)
    
    return tree

def _generate_CV_table(model_names, K):
    '''
    Generates a latex table with a row for each outer fold in the CV
    and a column for each coefficient and model test error in the CV
    Important, that baseline is first in model_names

    Input:
    :param file_names: A collection of model names for the models which have been tested in the CV
    :param K: How many folds there existed in the outer fold of the CV
    '''
    #Combine data for data frame
    D = np.ndarray((K,2*len(model_names)))
    D[:,0] = range(1,K+1)

    #Define columns for dataframe to latex table
    columns = ['Outer Fold']

    #Set columns and merge data
    for idx, m in enumerate(model_names):
        if m is not 'baseline':
            columns.append('{} Coef'.format(m))
            D[:,idx*2+1] = np.round(_load_npy('{}_lambdas'.format(m)),2)
            columns.append('{} Ei'.format(m))
            D[:,idx*2+2] = np.round(_load_npy(m),2)
        else:
            #no coefficients on the baseline model
            columns.append('{} Ei'.format(m))
            D[:,idx*2+1] = np.round(_load_npy(m),2)
        
    #Create data frame and convert to latex table
    df = pd.DataFrame(data=D, index = range(1,K+1), columns=columns)
    path = os.path.join(os.path.join(os.getcwd(),'william-logistic/cv_tables'),'CV_table.tex')
    _log(path, df.to_latex(index=False)+'\n\n')


def test_models(model_names):
    #Load standardized data
    X, y = cdc()

    #Outer and inner folds
    K_out = 10
    K_in = 10

    #regularization constants, using logspace to get small enough values
    lambdas = np.sort(np.logspace(-3, 2, 50))

    #Create models for validation
    logistic_regs = [logistic_reg(l) for l in lambdas]
    trees = [decision_tree(l) for l in lambdas]
    bases = [base_line for l in lambdas]

    E_gen_logi = nested_cv(X,y, logistic_regs, K_out, K_in, model_names[0], lambdas)
    E_gen_tree = nested_cv(X, y, trees, K_out, K_in, model_names[1], lambdas)
    E_gen_base = nested_cv(X, y, bases, K_out, K_in, model_names[2])

    print('E_gen_logi: {}'.format(E_gen_logi))
    print('E_gen_tree: {}'.format(E_gen_tree))
    print('E_gen_base: {}\n'.format(E_gen_base))

    _generate_CV_table(model_names, K_out)