"""
Christine Chung
ML for Public Policy
Spring 2016
Assignment 2
Data from: https://www.kaggle.com/c/GiveMeSomeCredit/data
"""

import explore
import process
import features
import build
import pandas as pd
from pipeline import Pipeline

PATH = 'cs-training.csv'
TEST = explore.snake_columns(pd.read_csv('cs-test.csv', index_col=0))
my_pipeline = Pipeline(PATH, TEST)

# explore
print('exploring')
my_pipeline.explore()

# add features
print('adding features')
#my_pipeline.add_col('monthly_income')
my_pipeline.add_col('age', transform_type='squared')
my_pipeline.create_binary('number_of_dependents')
my_pipeline.split('serious_dlqin2yrs')

# process
print('processing')
my_pipeline.impute('number_of_dependents', method='mode')
my_pipeline.impute('monthly_income', method='median')

my_pipeline.export_train_tocsv()

# build models
#['KNN','RF','LR','ET','AB','GB','NB','DT']
models=['KNN','DT','NB','LR']
#get X and y
print('building models')
my_pipeline.run_loops(models, 'testfile')
