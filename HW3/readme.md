Assignment 3 for Machine Learning in Public Policy, Spring 2016

Data From: https://www.kaggle.com/c/GiveMeSomeCredit/data

This repo contains:

Python Files:

explore: provides methods to reads and explore data, create histograms and html tables of summary statistics
process: provides methods to delete columns, observations, and impute data
features: provides methods to add features, log and square columns, split data into training and testing set
build: provides methods for using and evaluating different classifiers.
pipeline: provides a class to run a dataset through different models.
assignment3_pipeline: instantiates a pipeline for this assignment and runs credit data through pipeline.

PDF:

Report summarizing best classifiers.

Histograms of Features
CSVs:
cs-test: Provided by Kaggle. Used for predictions.
cs-training: Provided by Kaggle. Used for training.
imputed_data: cs-training post imputation
training: training data post split, imputation, and feature generation.
