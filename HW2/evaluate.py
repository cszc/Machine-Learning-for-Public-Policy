'''
6. Evaluate Classifier: you can use any metric you choose for this assignment
(accuracy is the easiest one)
'''
scores = cross_validation.cross_val_score(
...    clf, iris.data, iris.target, cv=5)

cross_validation.cross_val_score(model, X, y, scoring='accuracy')
