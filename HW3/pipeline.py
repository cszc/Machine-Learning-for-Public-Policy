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
        self.imputations = []
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.models = []


#exploring methods
    def explore(self):
        '''
        Explore summarizes data, snakes column names, and sets self.train to
        a dataframe created from the path to the unsplit training set.
        Does not update self.test
        '''
        self.update(explore.go(self.path))


#processing methods
    def impute(self, col_name, atype='mean', value=None, method=None):
        '''
        note: x_train needs to already be defined.
        note: only atype, value, OR method should be defined. throw and exception
        if more than one is specified.
        '''
        if self.x_train is None:
            #note: turn this into try except
            print('Need to split data set first.')
            return

        if not value:
            value = process.get_average(self.x_train, col_name, atype=atype)
        self.imputations.append((col_name, value))

        self.x_train = process.fill_missing(
            self.x_train, col_name, value=value, method=method)
        self.x_test = process.fill_missing(
            self.x_test, col_name, value=value, method=method)

    def add_col(self, col, transform_type='log'):
        '''
        '''
        self.update(
            (features.add_col(self.df, col, transform_type),
            features.add_col(self.test, col, transform_type)))

    def remove_col(self, col):
        '''
        '''
        del self.df[col]
        del self.test[col]

    def bin(self, col, mybins=10):
        '''
        '''
        self.update(
            (features.bin(self.df, col, mybins),
            features.bin(self.test, col, mybins)))

    def create_binary(self, col_name):
        '''
        '''
        series = self.df[col_name]
        test_series = self.test[col_name]
        self.update(
            (features.create_binary(self.df, series),
            features.create_binary(self.test, test_series)))

    def update(self, data_frames):
        '''
        Given two data frames in a list, updates the data training and testing
        data frames.
        '''
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
    def run_loops(self, models_to_run, filename):
        datasets = self.x_train, self.x_test, self.y_train, self.y_test
        build.run_loops(models_to_run, datasets, filename)

    def build(self, model_type):
        self.models.append(build.build_model(
            self.x_train, self.y_train, self.x_test, self.y_test, self.test, model_type))
