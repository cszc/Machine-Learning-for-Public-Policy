'''
Builds Classifiers and parameters
'''

from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import matplotlib
import json
matplotlib.style.use('ggplot')
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler


def define_clfs_params():
    clfs = {
            'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'ET': ExtraTreesClassifier(
                        n_estimators=10,
                        n_jobs=-1,
                        criterion='entropy'),
            'AB': AdaBoostClassifier(
                        DecisionTreeClassifier(max_depth=1),
                        algorithm="SAMME",
                        n_estimators=200),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
            'GB': GradientBoostingClassifier(
                        learning_rate=0.05,
                        subsample=0.5,
                        max_depth=6,
                        n_estimators=10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss="hinge", penalty="l2"),
            'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    grid = {
            'RF': {
                    'n_estimators': [1,10,100,1000,10000],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]
                    },
            'LR': {
                    'penalty': ['l1','l2'],
                    'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]
                    },
            'SGD': {
                    'loss': ['hinge','log','perceptron'],
                    'penalty': ['l2','l1','elasticnet']
                    },
            'ET': {
                    'n_estimators': [1,10,100,1000,10000],
                    'criterion' : ['gini', 'entropy'],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]
                    },
            'AB': {
                    'algorithm': ['SAMME', 'SAMME.R'],
                    'n_estimators': [1,10,100,1000,10000]
                    },
            'GB': {
                    'n_estimators': [1,10,100,1000,10000],
                    'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
                    'subsample' : [0.1,0.5,1.0],
                    'max_depth': [1,3,5,10,20,50,100]
                    },
            'NB' : {},
            'DT': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]
                    },
            'SVM' :{
                    'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
                    'kernel':['linear']
                    },
            'KNN' :{
                    'n_neighbors': [1,5,10,25,50,100],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto','ball_tree','kd_tree']
                    }
           }
    return clfs, grid


def clf_loop(models_to_run, clfs, params, datasets, filename):
    '''
    '''
    X_train, X_test, y_train, y_test = datasets
    comparison_metrics = {}
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        start = time.time()
        try:
            param_values = params[models_to_run[index]]
            model = GridSearchCV(clf, param_grid = param_values)
            model.fit(X_train, y_train)
            end = time.time()
            y_predictions = model.predict(X_test)
            metrics = evaluation_dict(y_test, y_predictions, (end - start))
            #Prints the parameters and eval metrics of the current model to screen
            print(model, '\n')
            print(json.dumps(metrics, indent = 2), '\n')
            comparison_metrics = build_comparison(comparison_metrics, metrics, models_to_run[index])
        except:
            print('Unexpected error raised for: ', str(clf))
    print(json.dumps(comparison_metrics, indent = 2), '\n')
    with open('comparison_metrics.txt', 'a') as outfile:
         json.dumps(comparison_metrics, outfile, indent = 2)
    plot_metrics(comparison_metrics, filename)


def run_loops(models_to_run, datasets, filename):
    clfs, params = define_clfs_params()
    clf_loop(models_to_run, clfs, params, datasets, filename)


def evaluation_dict(y_test, y_test_pred, run_time):
    '''Creates a dictionary with 5 prediction evaluation metrics and a measure
    of how long it took the model to run.'''
    results = {}
    metrics = {'Accuracy': accuracy_score, 'F1_Score': f1_score,
                'Precision': precision_score, 'Recall': recall_score,
                'AUC': roc_auc_score}
    for label, fn in metrics.items():
        results[label] = round(fn(y_test, y_test_pred), 4)
    results['Train Time (s)'] = run_time
    return results


def build_comparison(comparison_dict, model_dict, model_label):
    '''Adds the evaluation metrics of each model type to the comparison dictionary.
    Inputs the current comparison_dict, the dictionary of metrics for the current
    model, and the model label.'''
    for metric, value in model_dict.items():
        if metric in comparison_dict:
            comparison_dict[metric][model_label] = value
        else:
            comparison_dict[metric] = {}
            comparison_dict[metric][model_label] = value
    return comparison_dict


def write_to_csv(comparison_dict):
    #finish adding this feature
    x = json.loads(comparison_dict)
    f = csv.writer(open("test.csv", "a"))

    # Write CSV Header, If you dont need that, remove this line
    f.writerow(["Model", "Train_Time", "Accuracy", "F1_Score", "Precision", "Recall", "AUC"])


def plot_metrics(comparison_dict, filename):
    '''Plots metrics to compare models. Takes the comparison dictionary built
    while running models.'''
    f, axarr = plt.subplots(3, 2, figsize = (15,15))
    plt.setp(axarr)
    i = 0
    j = 0
    X = []
    for metric, models in comparison_dict.items():
        xlabels = models.keys()
        X = range(len(models))
        axarr[i, j].bar(X, models.values(), align='center', width=0.5)
        axarr[i, j].set_title(metric)
        i += 1
        if i == 3:
            j += 1
            i = 0
        plt.setp(axarr, xticks = X, xticklabels = xlabels)
        plt.savefig(filename)

#below not currently used
def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Not currently used
    '''
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    name = model_name
    plt.title(name)
    plt.savefig(name)


def precision_at_k(y_true, y_scores, k):
    '''
    not currently used
    '''
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)


def build_model(train_x, train_y, test_x, test_y, test, model_type='logistic'):
    '''
    Currently allows classifiers logistic and k-nearest neighbors.
    Takes in a set to train and test on, as well as a test dataset for prediction.
    Default model type is logistic.
    '''
    if model_type == 'logistic':
        train_x = train_x.select_dtypes(include=[numpy.number])
        test_x = test_x.select_dtypes(include=[numpy.number])
        clf = LogisticRegression()
        model = clf.fit(train_x, train_y)
        score = clf.score(test_x, test_y)
        predictions = clf.predict(test)
        probabilities = clf.predict_proba(test)[::,1]
        test['predicted'] = predictions
        test['probabilities'] = probabilities
        test.to_csv(model_type + '_predictions.csv')
        coef = clf.coef_
        return (model_type, model, predictions, coef, score)

    elif model_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=13)
        model = clf.fit(train_x, train_y)
        score = clf.score(test_x, test_y)
        return (model_type, model, score)

    else:
        print('Classifier not included yet')
