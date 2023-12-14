#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

params_sgd = {
    'loss': ['hinge'],
    'penalty': ['l2'],
    'max_iter': [10000],
    'alpha': [1e-4, 1e-3, 1e-2, 0.05, 0.1],
    'class_weight': ['balanced'],
    'fit_intercept': [False],
    'features_cutoff': [100, 1000]
}

param_grid_rf = {
    'max_depth': [1, 3, 10, 25, 100],
    'max_features': [1, 3, 10, 50, 100],
    'min_samples_split': [2, 3, 5, 10],
    'n_estimators': [1000],
    'criterion': ['gini'],
    'min_samples_leaf': [1, 3, 5, 10],
    'oob_score': [False],
    'class_weight': ['balanced'],
    'features_cutoff': [100, 1000]
}

param_grid_rfr = {
    'max_depth': [300],
    'max_features': [100],
    'min_samples_split': [300],
    'n_estimators': [100],
    'min_samples_leaf': [3],
    'oob_score': [False],
    'features_cutoff': [100, 1000]
}
param_grid_lda = {
    "solver": ["svd", "lsqr", "eigen"],
    'features_cutoff': [100, 1000]
}
param_grid_qda = {
}
param_grid_logreg = {
    # 'max_iter': [10000],
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'C': [20],
    'fit_intercept': [True],
    'max_iter': [50],
    'class_weight': ['balanced'],
    'features_cutoff': [100, 1000]
}
param_grid_linsvc = {
    'max_iter': [10000],
    'penalty': ["l1", "l2"],
    'C': [1, 3, 5, 7],
    'class_weight': ['balanced'],
    'features_cutoff': [100, 1000]
}
param_grid_svc = {
    'max_iter': [10000],
    'C': [0.5, 1, 3, 5],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'probability': [True],
    'features_cutoff': [100, 1000]

}
param_grid_ada = {
    'base_estimator': [LinearSVC(max_iter=10000)],
    'learning_rate': (1),
    'features_cutoff': [100, 1000]
}
param_grid_bag = {
    'base_estimator': [
        RandomForestClassifier(
            max_depth=50,
            max_features=1,
            min_samples_split=2,
            n_estimators=1000,
            criterion='gini',
            min_samples_leaf=1,
            oob_score=False,
            class_weight='balanced'
        )],
    'n_estimators': [100],
    'features_cutoff': [100, 1000]

}

param_grid_voting = {
    'voting': ('soft', 'hard'),
    'features_cutoff': [100, 1000]
}

param_grid_kn = {
    'features_cutoff': [100, 1000]
}

rf = RandomForestClassifier(max_depth=300,
                            max_features=100,
                            min_samples_split=300,
                            n_estimators=100)
gnb = GaussianNB()
cnb = CategoricalNB()
lr = LogisticRegression(max_iter=4000)
lsvc = LinearSVC(max_iter=10000)
estimators_list = [('rf', rf),
                   ('lr', lr),
                   ('lsvc', lsvc),
                   ('gnb', gnb),
                   ]

models = {
    "KNeighbors": [KNeighborsClassifier, param_grid_kn],
    # "LinearSVC": [LinearSVC, linsvc_space],
    # "SVCLinear": [SVC, svc_space],
    # "LDA": [LinearDiscriminantAnalysis, lda_space],
    # "LogisticRegression": [LogisticRegression, logreg_space],
    "RandomForestClassifier": [RandomForestClassifier, param_grid_rf],
    # "Gaussian_Naive_Bayes": [GaussianNB, nb_space],
    # "QDA": [QuadraticDiscriminantAnalysis, qda_space],
    # "SGDClassifier": [SGDClassifier, sgd_space],
    # "BaggingClassifier": [BaggingClassifier, bag_space],
    # "AdaBoost_Classifier": [AdaBoostClassifier, param_grid_ada],
    # "Voting_Classifier": [VotingClassifier, param_grid_voting],
}
