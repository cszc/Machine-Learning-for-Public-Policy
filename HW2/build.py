'''
Build Classifier: For this assignment, select any classifer you feel
comfortable with (Logistic Regression for example)
'''

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy
import pandas as pd

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
