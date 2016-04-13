'''
Generate Features: Write a sample function that can discretize a
continuous variable and one function that can take a categorical
variable and create binary variables from it.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


def bin(df, col_name, mybins):
    '''
    Takes a dataframe, column to bin, and number of bins.
    Returns a dataframe with the new column.
    '''
    df[col_name + '_bins'] = pd.cut(df[col_name], bins=mybins)
    return df


def create_binary(df, series):
    '''
    Takes a dataframe, and a columns as a series.
    Creates a set of columns as categorical variables based on the series.
    Returns the dataframe with the new columns.
    '''
    dummies = pd.get_dummies(series)
    for col_name in dummies.columns:
        df[col_name] = dummies[col_name]
    return df


def add_col(df, col, transform_type='log'):
    '''
    Adds new columns to a dataframe. Current options are log and squared.
    Returns dataframe with new column.
    '''
    if transform_type == 'log':
        #+1 to avoid taking log of 0
        df['log_'+col] = np.log(df[col] + 1)
    elif transform_type == 'squared':
        df[col+'^2'] = df[col]**2
    else:
        print('type not supported')
    return df


def split(df, depvar, test_size=0.15):
    '''
    Splits data into testing and training sets.
    Default is .15 testing, .85 training.
    Returns a list of 4 dataframes containing:
    train_test, train_y, train_test, train_y.
    Uses sklearn.cross_validation.train_test_split
    '''
    y = df[depvar]
    X =  df.drop(depvar, axis=1)
    return train_test_split(X, y, test_size=test_size)


# To come. Code from: https://github.com/yhat/DataGotham2013/blob/master/notebooks/7%20-%20Feature%20Engineering.ipynb
# def get_important_features(features, dependent_var):
#     clf = RandomForestClassifier(compute_importances=True)
#     clf.fit(df[features], df[dependent_var])
#     importances = clf.feature_importances_
#     sorted_idx = np.argsort(importances)
#     best_features = features[sorted_idx][::-1]
#     best_features
