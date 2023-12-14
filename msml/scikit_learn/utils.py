#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, Normalizer, StandardScaler, \
    PowerTransformer, QuantileTransformer, PolynomialFeatures, Binarizer
from sklearn.pipeline import Pipeline

from msml.utils.utils import plot_confusion_matrix, get_unique_labels


def get_labels(fname):
    """
    Gets labels

    :param fname:
    :return:
    """
    meta = pd.read_excel(fname, header=0)
    toremove = pd.isnull(meta.values[:, 0])
    tokeep = [i for i, x in enumerate(toremove) if x == 0]

    meta = meta.iloc[tokeep, :]
    samples_classes = meta['Pathological type']
    classes = np.unique(samples_classes)

    return classes, samples_classes


def split_labels_indices(labels, train_inds):
    """
    Splits labels indices

    :param labels:
    :param train_inds:
    :return:
    """
    train_indices = []
    test_indices = []
    for j, sample in enumerate(list(labels)):
        if sample in train_inds:
            train_indices += [j]
        else:
            test_indices += [j]

    assert len(test_indices) != 0
    assert len(train_indices) != 0

    return train_indices, test_indices


def split_train_test(labels):
    """
    Splits labels into train and tests

    :param labels:
    :return:
    """

    # First, get all unique samples and their category
    unique_samples = np.arange(0, len(labels))

    # StratifiedKFold with n_splits of 5 to ranmdomly split 80/20.
    # Used only once for train/tests split.
    # The train split needs to be split again into train/valid sets later
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    train_inds, test_inds = next(skf.split(np.zeros_like(labels), labels))
    train_cats = [labels[ind] for ind in train_inds]

    assert len(unique_samples) == len(train_inds) + len(test_inds)
    assert len([x for x in test_inds if x in train_inds]) == 0
    assert len(np.unique([labels[ind] for ind in test_inds])) > 1

    return train_inds, test_inds, train_cats


def get_scaler(scaler_str):
    """
    Returns the scaler corresponding to the string received as input

    :param scaler_str:
    :return:
    """
    if str(scaler_str) == 'binarize':
        scale = Pipeline([('minmax', MinMaxScaler()), ('binarize', Binarizer(threshold=0.9))])
    elif str(scaler_str) == 'normalizer':
        scale = Normalizer
    elif str(scaler_str) == 'standard':
        scale = StandardScaler
    elif str(scaler_str) == 'minmax':
        scale = MinMaxScaler
    elif str(scaler_str) == 'maxabs':
        scale = MaxAbsScaler
    elif str(scaler_str) == "robust":
        scale = RobustScaler
    elif str(scaler_str) == 'quantiles':
        scale = QuantileTransformer
    elif str(scaler_str) == 'polynomial':
        scale = PolynomialFeatures
    elif str(scaler_str) == 'power':
        scale = PowerTransformer
    elif str(scaler_str) == "spline":
        from sklearn.preprocessing import SplineTransformer
        scale = SplineTransformer
    else:
        sys.exit(f"Invalid scaler {scaler_str}")
    return scale


def load_data(path, drop_l, drop_b, binary, min_rt, min_mz):
    """
    This function is much faster than using pd.read_csv

    :param path:
    :param drop_l:
    :param drop_b:
    :param binary:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as csv_file:
        rows = csv.DictReader(csv_file)
        data = []
        labels = []
        row = {}
        for i, row in enumerate(rows):
            rts_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if int(x.split("_")[2]) > min_rt]
            mzs_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if float(x.split("_")[3]) > min_mz]
            to_keep = np.intersect1d(rts_to_keep, mzs_to_keep)
            labels += [list(row.values())[0]]
            data += [np.array(list(row.values())[1:], dtype=float)[to_keep]]

    data = np.stack(data)

    data[np.isnan(data)] = 0
    dframe = pd.DataFrame(data, index=labels, columns=np.array(list(row.keys()))[1:][to_keep])
    if drop_l:
        dframe = drop_lows(dframe)
        data = dframe.values
        labels = dframe.index

    if drop_b:
        dframe = drop_blks(dframe)
        data = dframe.values
        labels = dframe.index
    b_list = ["kox", "sau", "blk", "pae", "sep"]
    batches = np.array(
        [0 if x.split('_')[0] in b_list and 'blk_p' not in x else 1 for x in labels])
    lows = np.array([1 if 'l' in d.split('_')[1] else 0 for d in labels])
    labels = np.array([d.split('_')[0].split('-')[0] for d in labels])
    if binary:
        for i, label in enumerate(labels):
            if label != 'blk':
                labels[i] = 'not_blk'
            else:
                labels[i] = 'blk'
    classes = np.array(
        [np.argwhere(label == get_unique_labels(labels))[0][0] for label in labels]
    )

    return {
        "data": data,
        "labels": labels,
        "batches": batches,
        "classes": classes,
        "lows": lows,
    }


def load_data2(path, drop_l, drop_b, binary, min_rt, min_mz):
    """
    This function is much faster than using pd.read_csv

    :param path:
    :param drop_l:
    :param drop_b:
    :param binary:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as csv_file:
        rows = csv.DictReader(csv_file)
        data = []
        labels = []
        row = {}
        for i, row in enumerate(rows):
            rts_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if int(x.split("_")[2]) > min_rt]
            mzs_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if float(x.split("_")[3]) > min_mz]
            to_keep = np.intersect1d(rts_to_keep, mzs_to_keep)
            labels += [list(row.values())[0]]
            data += [np.array(list(row.values())[1:], dtype=float)[to_keep]]
    data = np.stack(data)

    data[np.isnan(data)] = 0
    dframe = pd.DataFrame(data, index=labels, columns=np.array(list(row.keys()))[1:][to_keep])
    if drop_l:
        dframe = drop_lows(dframe)
        data = dframe.values
        labels = dframe.index

    if drop_b:
        dframe = drop_blks(dframe)
        data = dframe.values
        labels = dframe.index

    batches = np.array([d.split('_')[0] for d in labels])
    lows = np.array([1 if 'l' in d.split('_')[1] else 0 for d in labels])
    labels = np.array([d.split('_')[1].split('-')[0] for d in labels])
    if binary:
        for i, label in enumerate(labels):
            if label != 'blk':
                labels[i] = 'not_blk'
            else:
                labels[i] = 'blk'
    classes = np.array(
        [np.argwhere(label == get_unique_labels(labels))[0][0] for label in labels]
    )

    return {
        "data": data,
        "labels": labels,
        "batches": batches,
        "classes": classes,
        "lows": lows,
    }


def get_estimators_list():
    """Gets a list of classifiers for VoterClassifier"""
    rfc = RandomForestClassifier(max_depth=300,
                                 max_features=100,
                                 min_samples_split=300,
                                 n_estimators=100)
    gnb = GaussianNB()
    logreg = LogisticRegression(max_iter=4000)
    lsvc = SVC(kernel='linear', probability=True)
    estimators_list = [('gnb', gnb),
                       ('rfc', rfc),
                       ('lr', logreg),
                       ('lsvc', lsvc)
                       ]
    return estimators_list


def get_confusion_matrix(reals, preds, unique_labels):
    acc = np.mean([1 if pred == label else 0 for pred, label in zip(preds, reals)])
    cm = metrics.confusion_matrix(reals, preds)
    figure = plot_confusion_matrix(cm, unique_labels, acc)

    # cm = np.zeros([len(unique_labels), len(unique_labels)])
    # for real, pred in zip(reals, preds):
    #     confusion_matrix[int(real), int(pred)] += 1
    # indices = [f"{lab}" for lab in unique_labels]
    # columns = [f"{lab}" for lab in unique_labels]
    return figure


def save_confusion_matrix(fig, name, acc):
    # sns_plot = sns.heatmap(df, annot=True, square=True, cmap="YlGnBu",
    #                        annot_kws={"size": 35 / np.sqrt(len(df))})
    # fig = sns_plot.get_figure()
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    plt.title(f'Confusion Matrix (acc={np.round(acc, 3)})')
    os.makedirs(f'{dirs}/', exist_ok=True)
    stuck = True
    while stuck:
        try:
            fig.savefig(f"{dirs}/cm_{name}.png")
            stuck = False
        except:
            print('stuck...')
    plt.close()


def save_roc_curve(model, x_test, y_test, unique_labels, name, binary, acc):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_score = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC_{name}.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        y_pred_proba = model.predict_proba(x_test)
        y_preds = model.predict(x_test)
        n_classes = len(unique_labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        bin_label = label_binarize(y_test, classes=classes)
        roc_score = roc_auc_score(y_true=label_binarize(y_test, classes=classes[bin_label.sum(0) != 0]),
                                  y_score=label_binarize(y_preds, classes=classes[bin_label.sum(0) != 0]),
                                  multi_class='ovr')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=classes)[:, i], y_pred_proba[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # roc for each class
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC={np.round(roc_score, 3)}, acc={np.round(acc, 3)})')
        # ax.plot(fpr[0], tpr[0], label=f'AUC = {np.round(roc_score, 3)} (All)', color='k')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'AUC = {np.round(roc_auc[i], 3)} ({unique_labels[i]})')
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        # sns.despine()
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC_{name}.png')
                stuck = False
            except:
                print('stuck...')

        plt.close()

    return roc_score


def drop_lows(train_data):
    inds = train_data.index
    for ind in inds:
        if 'l' in ind.split('_')[1]:
            train_data = train_data.drop(ind)
    return train_data


# TODO put this function in utils
def drop_blks(train_data):
    inds = train_data.index
    for ind in inds:
        if 'blk' in ind:
            train_data = train_data.drop(ind)
    return train_data

