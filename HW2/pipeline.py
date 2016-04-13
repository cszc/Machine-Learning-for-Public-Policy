import explore
import process
import features
import build
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

class Pipeline:
    '''
    Pipeline class takes a path to a dataset, and a testing dataframe.
    '''
    def __init__(self, path, test):
        self.path = path
        self.test = test
        self.df = explore.read_in(path)
        self.features = None
        self.depvar = None
        self.unused_features = []
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.models = []

#exploring methods
    def explore(self):
        self.update(explore.go(self.path))

#processing methods
    def impute(self, col_name, method='mean', value=None):
        self.update(
            (process.fill_missing(self.df, col_name, method, value),
            process.fill_missing(self.test, col_name, method, value)))

    def add_col(self, col, transform_type='log'):
        self.update(
            (features.add_col(self.df, col, transform_type),
            features.add_col(self.test, col, transform_type)))

    def remove_col(self, col):
        del self.df[col]
        del self.test[col]

    def bin(self, col, mybins=10):
        self.update(
            (features.bin(self.df, col, mybins),
            features.bin(self.test, col, mybins)))

    def create_binary(self, col_name):
        series = self.df[col_name]
        test_series = self.test[col_name]
        self.update(
            (features.create_binary(self.df, series),
            features.create_binary(self.test, test_series)))

    def update(self, data_frames):
        train = 0
        test = 1
        self.df = data_frames[train]
        if data_frames[test] is not None:
            self.test = data_frames[test]

    def split(self, depvar):
        self.test = self.test.drop(depvar, 1)
        self.depvar = depvar
        self.x_train, self.x_test, self.y_train, self.y_test = features.split(
            self.df, depvar, test_size=0.15)
        self.features = self.x_train.columns

    def export_train_tocsv(self):
        self.x_train.to_csv('training.csv')


#building methods
    def build(self, model_type):
        self.models.append(build.build_model(
            self.x_train, self.y_train, self.x_test, self.y_test, self.test, model_type))
