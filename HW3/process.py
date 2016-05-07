"""
This process module (2/5) pre-processes data. It provides functions for
imputing missing values and splitting data sets.
"""

from sklearn.cross_validation import train_test_split
import pandas as pd
import json
import requests


def predict_gender(df):
    '''
    Imputes missing gender with predicted gender using genderize.io
    '''
    for i, frame in df['Gender'].iteritems():
        if pd.isnull(frame):
            name = df.xs(i)['First_name']
            r = requests.get('https://api.genderize.io/?name=' + name)
            gender = r.json()['gender']
            df.set_value(i, 'Gender', gender)
    return df


def get_average(df, col_name, atype='mean'):
    if atype == 'mean':
        average = round(df[col_name].mean())
    elif atype == 'median':
        average = df[col_name].median().iloc[0]
    elif atype == 'mode':
        average = df[col_name].mode().iloc[0]
    return average


def fill_missing(df, col_name, value=None, method='back'):
    '''
    Imputes missing data using specified method. Available
    methods are: mean, mode, back fill, and forward fill.
    If a value is provided instead, imputes with the given
    value.
    Returns the imputed dataframe and writes the new
    dataframe out to a csv.
    '''
    if value:
        df[col_name].fillna(value=value, inplace=True)
        df.to_csv('imputed_data.csv')
        return df

    if method == 'back':
        df.bfill()
    elif method == 'forward':
        df.ffill()
    else:
        print('Method does not exist')
    df.to_csv('imputed_data.csv')
    return df


def conditional_impute(df, columns, classifier):
    #for the future
    return
