#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import warnings
import pickle

import numpy as np
import pandas as pd
import os
import csv
import json
import matplotlib

matplotlib.use('Agg')
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold, RepeatedKFold, \
    StratifiedShuffleSplit
from msml.scikit_learn.utils import get_scaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef as MCC
from combat.pycombat import pycombat
from skopt import gp_minimize
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from msml.scikit_learn.train.params_gp import *
from msml.utils.utils import plot_confusion_matrix

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src/models/sklearn/'


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


# m, unique_labels, binary_model, x_valid, y_valid, x_valid_binary, y_valid_binary, 10000, n_features_binary
def hierarchical_predictions(m, unique_labels, binary_model, x_valid, y_valid, x_valid_binary, y_valid_binary,
                             n_features, n_features_binary):
    try:
        y_valid_preds = binary_model.predict(x_valid_binary)
    except:
        pass
    blk_index = np.argwhere(unique_labels == 'blk')[0][0]

    x_valid_scores = np.array([1 if y == pred else 0 for pred, y in zip(y_valid_preds, y_valid_binary)])
    blks_indices = np.argwhere(y_valid_preds == blk_index).squeeze(1)
    not_blks_indices = [i for i in range(len(y_valid_binary)) if i not in blks_indices]

    try:
        assert len(x_valid_scores) == len(not_blks_indices) + len(blks_indices)
    except:
        pass
    blk_valid_score = np.sum(x_valid_scores[blks_indices])
    # for validation, samples that are wrongfully classified as blks are not included
    # We exclude all samples predicted as
    x_valid_not_blks = x_valid[not_blks_indices]  # [:, :n_features]
    y_valid_not_blks = y_valid[not_blks_indices]
    # c_valid_not_blks = pd.Categorical(y_valid_not_blks).codes

    try:
        assert len(np.unique(y_valid_not_blks)) == len(np.unique(y_valid)) - 1
    except:
        pass
    assert len(blks_indices) + len(not_blks_indices) == len(y_valid)
    valid_preds_not_blks = m.predict(x_valid_not_blks)

    valid_score_not_blks = np.array(
        [1 if y == pred else 0 for pred, y in zip(valid_preds_not_blks, y_valid_not_blks)])

    valid_score = (blk_valid_score + valid_score_not_blks.sum()) / len(x_valid)

    valid_preds_blks = y_valid_preds[blks_indices]
    y_valid_blks = y_valid_binary[blks_indices]

    return valid_score, blk_valid_score, valid_score_not_blks, blks_indices, valid_preds_blks, y_valid_blks, valid_preds_not_blks, y_valid_not_blks


# TODO put this function in utils
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


# TODO put this function in utils
# This function is much faster than using pd.read_csv
def load_data(path, drop_l, drop_b, binary, min_rt, min_mz):
    rows = csv.DictReader(open(path))
    data = []
    labels = []
    for i, row in enumerate(rows):
        rts_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if int(x.split("_")[2]) >= min_rt]
        mzs_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if float(x.split("_")[3]) >= min_mz]
        rts_to_keep.extend(mzs_to_keep)
        to_keep = np.unique(np.array(rts_to_keep))
        labels += [list(row.values())[0]]
        data += [np.array(list(row.values())[1:], dtype=float)[to_keep]]

    lows = np.array([1 if 'l' in d.split('_')[1] and 'blk' not in d else 0 for d in labels])
    data = np.stack(data)

    data[np.isnan(data)] = 0
    print(data.shape)

    df = pd.DataFrame(data, index=labels, columns=np.array(list(row.keys())[1:])[to_keep])

    labels = np.array([d.split('_')[0].split('-')[0] for d in labels])
    if drop_l:
        df = drop_lows(df)
        # data = df.values
        labels = df.index

    if drop_b:
        df = drop_blks(df)
        # data = df.values
        labels = df.index

    batches = np.array(
        [0 if x.split('_')[0] in ["kox", "sau", "blk", "pae", "sep"] and 'blk_p' not in x else 1 for x in labels])
    labels = np.array([d.split('_')[0].split('-')[0] for d in labels])
    if binary:
        for i, label in enumerate(labels):
            if label != 'blk':
                labels[i] = 'not_blk'
            else:
                labels[i] = 'blk'

    return df, labels, batches, lows


def new_load_data(path, binary=0, min_rt=0, min_mz=0, valid_batch=0):
    rows = csv.DictReader(open(path))
    data = []
    labels = []
    for i, row in enumerate(rows):
        rts_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if float(x.split("_")[1]) >= min_rt]
        mzs_to_keep = [i for i, x in enumerate(list(row.keys())[1:]) if float(x.split("_")[2]) >= min_mz]
        rts_to_keep.extend(mzs_to_keep)
        to_keep = np.unique(np.array(rts_to_keep))
        labels += [list(row.values())[0]]
        data += [np.array(list(row.values())[1:], dtype=float)[to_keep]]

    data = np.stack(data)

    data[np.isnan(data)] = 0
    print(data.shape)

    df = pd.DataFrame(data, index=labels, columns=np.array(list(row.keys())[1:])[to_keep])

    if valid_batch != 1:
        batches = np.array([x.split('_')[0] for x in labels])
    else:
        batches = np.array(['valid' for x in labels])

    lows = np.array([1 if 'l' in d.split('_')[2] else 0 for d in labels])
    labels = np.array([d.split('_')[1].split('-')[0] for d in labels])
    if binary:
        for i, label in enumerate(labels):
            if label != 'blk':
                labels[i] = 'not_blk'
            else:
                labels[i] = 'blk'

    return df, labels, batches, lows


class Train:
    def __init__(self, model, data, hparams_names, args, ovr, binary=True):
        self.best_roc_score = -1
        try:
            with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
                self.previous_models = json.load(json_file)
        except:
            self.previous_models = {}
        self.ovr = ovr
        self.binary = binary
        self.random = args.random
        self.model = model
        self.data = data
        self.hparams_names = hparams_names
        # self.train_indices, self.test_indices, _ = split_train_test(self.labels)
        self.n_splits = args.n_splits

        self.n_repeats = args.n_repeats
        self.jackknife = args.jackknife
        self.best_scores_train = None
        self.best_scores_valid = None
        self.best_mccs_train = None
        self.best_mccs_valid = None
        self.scores_train = None
        self.scores_valid = None
        self.mccs_train = None
        self.mccs_valid = None
        self.y_preds = np.array([])
        self.y_valids = np.array([])
        self.top3_valid = None
        self.top3_train = None
        self.iter = 0

    def train(self, h_params):
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        for name, param in zip(self.hparams_names, h_params):
            if name == 'features_cutoff':
                features_cutoff = param
            elif name == 'threshold':
                threshold = param
            else:
                param_grid[name] = param
        try:
            assert features_cutoff is not None
        except AttributeError:
            exit('features_cutoff not in the hyperparameters. Leaving')

        # Find why test has 1 more column
        if self.data['train']['data'].shape[1] > self.data['test']['data'].shape[1]:
            self.data['train']['data'] = self.data['train']['data'][:, :self.data['test']['data'].shape[1]]
        elif self.data['test']['data'].shape[1] > self.data['train']['data'].shape[1]:
            self.data['test']['data'] = self.data['test']['data'][:, :self.data['train']['data'].shape[1]]
        # Remove features with a ratio of data > 0 that is < a threshold
        good_features = np.array(
            [i for i in range(self.data['train']['data'].shape[1]) if sum(self.data['train']['data'][:, i] != 0) > int(self.data['train']['data'].shape[0] * 0.1)])
        self.data['train']['data'] = self.data['train']['data'][:, good_features]
        self.data['test']['data'] = self.data['test']['data'][:, good_features]

        train_labels = self.data['train']['labels']
        train_data = self.data['train']['data']
        train_batches = self.data['train']['batches']

        test_labels = self.data['test']['labels']
        test_data = self.data['test']['data']
        test_batches = self.data['test']['batches']


        low_labels = self.data['lows']['labels']
        low_data = self.data['test']['data']
        low_batches = self.data['test']['batches']

        unique_labels = []
        for l in train_labels:
            if l not in unique_labels:
                unique_labels += [l]

        unique_labels = np.array(unique_labels)
        train_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in train_labels])
        test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in test_labels])
        if data["lows"]["data"] is not None:
            low_test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in low_labels])

        if self.n_splits != -1:
            if self.n_repeats > 1:
                if self.jackknife:
                    self.jackknife = False
                    self.n_repeats = 1
                    print('Jacknifing cannot be combined with Repeated Holdout. Doing Repeated Holdout.')
                skf = RepeatedStratifiedKFold(n_repeats=self.n_repeats, n_splits=self.n_splits)
            elif self.jackknife:
                skf = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2)
            else:
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        else:
            if self.jackknife:
                print('Jacknifing cannot be combined with LOOCV. Doing Repeated LOOCV')
                self.jackknife = False
            if self.n_repeats > 1:
                skf = RepeatedKFold(n_repeats=self.n_repeats, n_splits=len(self.train_indices))
            else:
                skf = KFold(n_splits=len(self.train_indices), shuffle=True, random_state=1)
        all_x_train = train_data
        all_batches_train = train_batches
        all_y_train = train_classes
        if args.scaler != 'binarize':
            scaler = get_scaler(args.scaler)()
        else:
            from sklearn.preprocessing import Binarizer
            scaler = Pipeline([('minmax', get_scaler('minmax')()), ('binarizer', Binarizer(threshold=threshold))])
        if test_data.shape[1] < 10000:
            cutoff = test_data.shape[1]
        else:
            cutoff = 10000
        if args.correct_batches:
            df_train = pd.DataFrame(all_x_train[:, :cutoff], index=all_y_train)
            df_test = pd.DataFrame(test_data[:, :cutoff], index=test_classes)
            df = pd.concat((df_train, df_test), 0).T
            all_batches = np.concatenate((all_batches_train, test_batches))
            if data["lows"]["data"] is not None:
                df_low_test = pd.DataFrame(data["lows"]["data"][:, :cutoff], index=low_test_classes)
                df = pd.concat((df, df_low_test.T), 1)
                all_batches = np.concatenate((all_batches, low_test_batches))
            df[df.isna()] = 0
            all_train = pycombat(df, all_batches).T.values
            all_x_train = all_train[:data["train"]["data"].shape[0]]

            del all_train

        print(f'Iteration: {self.iter}')

        self.scores_train = []
        self.scores_valid = []
        self.mccs_train = []
        self.mccs_valid = []
        self.top3_train = []
        self.top3_valid = []
        broke = False
        y_pred_valids = []
        y_valids = []
        x_valids = []
        train_nums = np.arange(0, len(all_x_train))
        for i, (train_inds, valid_inds) in enumerate(skf.split(train_nums, all_y_train)):
            # Just plot the first iteration, it will already be crowded if doing > 100 optimization iterations
            print(f"CV: {i + 1}")
            # new_nums = [train_indices[i] for i in inds]

            x_train = all_x_train[train_inds]
            feats_inds = np.arange(x_train.shape[1])

            feats_inds = feats_inds[:features_cutoff]
            x_train = x_train[:, feats_inds]
            y_train = all_y_train[train_inds]
            x_valid = all_x_train[valid_inds]
            x_valid = x_valid[:, feats_inds]
            y_valid = all_y_train[valid_inds]

            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)

            m = self.model()
            m.set_params(**param_grid)
            if self.ovr:
                m = OneVsRestClassifier(m)

            try:
                m.fit(x_train, y_train)
            except:
                break

            score_valid = m.score(x_valid, y_valid)
            score_train = m.score(x_train, y_train)
            print('valid_score:', score_valid, "features_cutoff", features_cutoff, 'h_params:', param_grid)
            self.scores_train += [score_train]
            self.scores_valid += [score_valid]
            y_pred_train = m.predict(x_train)
            y_pred_valid = m.predict(x_valid)
            y_pred_valids.extend(m.predict(x_valid))
            y_valids.extend(y_valid)
            x_valids.extend(x_valid)
            mcc_train = MCC(y_train, y_pred_train)
            mcc_valid = MCC(y_valid, y_pred_valid)

            self.mccs_train += [mcc_train]
            self.mccs_valid += [mcc_valid]

            try:
                y_proba_train = m.predict_proba(x_train)
                y_proba_valid = m.predict_proba(x_valid)
                y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                        zip(y_proba_train.argsort(1), y_train)])
                y_top3_valid = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                        zip(y_proba_valid.argsort(1), y_valid)])

                self.top3_train += [y_top3_train]
                self.top3_valid += [y_top3_valid]
            except:
                self.top3_train += [-1]
                self.top3_valid += [-1]

        if self.best_scores_valid is None:
            self.best_scores_valid = 0

        score = np.mean(self.scores_valid)
        if np.isnan(score):
            score = 0
        if model not in self.previous_models:
            self.previous_models[model] = {'valid_acc_mean': -1}
        if score > np.mean(self.best_scores_valid) and score > float(self.previous_models[model]['valid_acc_mean']):
            self.best_scores_train = self.scores_train
            self.best_scores_valid = self.scores_valid
            self.best_mccs_train = self.mccs_train
            self.best_mccs_valid = self.mccs_valid
            self.best_top3_train = self.top3_train
            self.best_top3_valid = self.top3_valid
            fig = get_confusion_matrix(y_pred_valids, y_valids, unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid", acc=score)
            try:
                self.best_roc_score = save_roc_curve(m, x_valids, y_valids, unique_labels,
                                                     f"{args.destination}/ROC/{model}_valid", binary=args.binary,
                                                     acc=score)
            except:
                pass
        return 1 - score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_calls', type=int, default=100)
    parser.add_argument('--ovr', type=int, default=1, help='OneVsAll strategy')
    parser.add_argument('--drop_lows', type=str, default="train")
    parser.add_argument('--drop_blks', type=int, default=0)
    parser.add_argument('--correct_batches', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--spd', type=int, default=300)
    parser.add_argument('--mz', type=float, default=0.1)
    parser.add_argument('--rt', type=float, default=10)
    parser.add_argument('--min_rt', type=float, default=60)
    parser.add_argument('--min_mz', type=float, default=200)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--run_name', type=str, default='4')
    parser.add_argument('--scaler', type=str, default='binarize')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--jackknife', type=str, default=0)
    parser.add_argument("--input_dir", type=str, help="Path to intensities csv file")
    parser.add_argument("--output_dir", type=str, help="Path to intensities csv file")
    args = parser.parse_args()

    if int(args.rt) == args.rt:
        args.rt = int(args.rt)
    if int(args.mz) == args.mz:
        args.mz = int(args.mz)
    args.destination = f"{args.output_dir}/mz{args.mz}/rt{args.rt}/minmz{args.min_mz}/minrt{args.min_rt}/{args.spd}spd/{args.scaler}/" \
                       f"corrected{args.correct_batches}/drop_lows{args.drop_lows}/drop_blks{args.drop_blks}/binary{args.binary}/" \
                       f"boot{args.jackknife}/{args.scaler}/cv{args.n_splits}/nrep{args.n_repeats}/ovr{args.ovr}/{args.run_name}/"

    # args.train_input_binary = f"{args.input_dir}/train/BACT_train_inputs_gt0.15_{args.run_name}.csv"
    # args.test_input_binary = f"{args.input_dir}/test/BACT_test_inputs_gt0.15_{args.run_name}.csv"
    args.train_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/train/{args.run_name}/BACT_train_inputs_gt0.15_{args.run_name}.csv"
    # args.test_input = f"{args.input_dir}/test/BACT_test_inputs_gt0.15_{args.run_name}.csv"

    args.inference_inputs = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/valid/{args.run_name}/BACT_inference_inputs_{args.run_name}.csv"

    models = {
        # "BaggingClassifier": [BaggingClassifier, bag_space],
        "KNeighbors": [KNeighborsClassifier, kn_space],
        "LinearSVC": [LinearSVC, linsvc_space],
        # "LogisticRegression": [LogisticRegression, logreg_space],
        "Gaussian_Naive_Bayes": [GaussianNB, nb_space],
        # "QDA": [QuadraticDiscriminantAnalysis, qda_space],
        "SGDClassifier": [SGDClassifier, sgd_space],
        "SVCLinear": [SVC, svc_space],
        "RandomForestClassifier": [RandomForestClassifier, rfc_space],
        # "LDA": [LinearDiscriminantAnalysis, lda_space],  # Creates an error...
        # "AdaBoost_Classifier": [AdaBoostClassifier, param_grid_ada],
        # "Voting_Classifier": [VotingClassifier, param_grid_voting],
    }

    try:
        with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
            previous_models = json.load(json_file)
    except:
        previous_models = {}
    low_train_data = None
    low_test_data = None
    if args.drop_lows == "train":
        train_data, train_labels, train_batches, train_lows = new_load_data(args.train_input, drop_l=False,
                                                                            drop_b=args.drop_blks, binary=args.binary,
                                                                            min_mz=args.min_mz, min_rt=args.min_rt)
        test_data, test_labels, test_batches, test_lows = new_load_data(args.test_input, binary=args.binary,
                                                                        min_mz=args.min_mz, min_rt=args.min_rt)
        assert train_data.columns == test_data.columns

        if args.binary:
            for i, label in enumerate(train_labels):
                if label != 'blk':
                    train_labels[i] = 'not'
            for i, label in enumerate(test_labels):
                if label != 'blk':
                    test_labels[i] = 'not'

        test_low_inds = np.arange(len(test_lows))[test_lows == 1]
        train_low_inds = np.arange(len(train_lows))[train_lows == 1]

        low_train_data = train_data[train_low_inds]
        low_train_labels = train_labels[train_low_inds]
        low_train_batches = train_batches[train_low_inds]

        low_test_data = test_data[test_low_inds]
        low_test_labels = test_labels[test_low_inds]
        low_test_batches = test_batches[test_low_inds]

        test_data = np.delete(test_data, test_low_inds, axis=0)
        test_labels = np.delete(test_labels, test_low_inds, axis=0)
        test_batches = np.delete(test_batches, test_low_inds, axis=0)
        train_data = np.delete(train_data, train_low_inds, axis=0)
        train_labels = np.delete(train_labels, train_low_inds, axis=0)
        train_batches = np.delete(train_batches, train_low_inds, axis=0)

        low_test_data = np.concatenate((low_train_data, low_test_data))
        low_test_labels = np.concatenate((low_test_labels, low_train_labels))
        low_test_batches = np.concatenate((low_test_batches, low_train_batches))

        low_train_data = None
        low_train_labels = None
        low_train_batches = None

    elif args.drop_lows == "test":
        train_data, train_labels, train_batches, train_lows = load_data(args.train_input, drop_l=False,
                                                                        drop_b=args.drop_blks, binary=args.binary,
                                                                        min_mz=args.min_mz, min_rt=args.min_rt)
        test_data, test_labels, test_batches, test_lows = load_data(args.test_input, drop_l=True,
                                                                    drop_b=args.drop_blks, binary=args.binary,
                                                                    min_mz=args.min_mz, min_rt=args.min_rt)

        low_inds = np.arange(len(test_lows))[test_lows == 1]
        train_data = np.concatenate((train_data, test_data[low_inds]))
        train_labels = np.concatenate((train_labels, test_labels[low_inds]))
        train_batches = np.concatenate((train_batches, test_batches[low_inds]))

        test_data = np.delete(test_data, low_inds, axis=0)
        test_labels = np.delete(test_labels, low_inds, axis=0)
        test_batches = np.delete(test_batches, low_inds, axis=0)

        low_test_data = None

    elif args.drop_lows == "all":
        train_data, train_labels, train_batches, train_lows = load_data(args.train_input, drop_l=True,
                                                                        drop_b=args.drop_blks, binary=args.binary,
                                                                        min_mz=args.min_mz, min_rt=args.min_rt)
        test_data, test_labels, test_batches, test_lows = load_data(args.test_input, drop_l=True,
                                                                    drop_b=args.drop_blks, binary=args.binary,
                                                                    min_mz=args.min_mz, min_rt=args.min_rt)
    else:
        test_data, test_labels, test_batches, test_lows = new_load_data(args.inference_inputs, binary=args.binary,
                                                                        min_mz=args.min_mz, min_rt=args.min_rt, valid_batch=1)
        train_data, train_labels, train_batches, train_lows = new_load_data(args.train_input, binary=args.binary,
                                                                            min_mz=args.min_mz, min_rt=args.min_rt)
        # train_data2, train_labels2, train_batches2, train_lows2 = new_load_data(args.test_input, binary=args.binary,
        #                                                                     min_mz=args.min_mz, min_rt=args.min_rt)
        # train_data = pd.concat((train_data, train_data2))
        # train_batches = np.concatenate((train_batches, train_batches2))
        # train_labels = np.concatenate((train_labels, train_labels2))
        # train_lows = np.concatenate((train_lows, train_lows2))

        test_low_inds = np.arange(len(test_lows))[test_lows == 1]
        train_low_inds = np.arange(len(train_lows))[train_lows == 1]

        low_test_data = test_data.iloc[test_low_inds]
        low_test_labels = test_labels[test_low_inds]
        low_test_batches = test_batches[test_low_inds]

        test_data = np.delete(test_data.values, test_low_inds, axis=0)
        test_labels = np.delete(test_labels, test_low_inds, axis=0)
        test_batches = np.delete(test_batches, test_low_inds, axis=0)

        train_data = train_data.values

    data = {
        'train': {
            'labels': train_labels,
            'data': train_data,
            'batches': train_batches,
            'lows': train_lows,
        },
        'test': {
            'labels': test_labels,
            'data': test_data,
            'batches': test_batches,
            'lows': test_lows,
        },
    }
    if low_test_data is not None:
        data['lows'] = {
            'labels': low_test_labels,
            'data': low_test_data.values,
            'batches': low_test_batches,
        }
    elif low_train_data is not None:
        data['lows'] = {
            'labels': low_train_labels,
            'data': low_train_data,
            'batches': low_train_batches,
        }
    else:
        data['lows'] = {
            'labels': None,
            'data': None,
            'batches': None,
        }

    unique_labels = []
    for l in train_labels:
        if l not in unique_labels:
            unique_labels += [l]

    unique_labels = np.array(unique_labels)
    train_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in train_labels])
    test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in test_labels])
    if low_test_data is not None:
        low_test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in low_test_labels])
    best_params_dict = previous_models
    os.makedirs(f"{args.destination}/saved_models/sklearn/", exist_ok=True)
    for model in models:
        print(f"Training {model}")
        hparams_names = [x.name for x in models[model][1]]
        train = Train(models[model][0], data, hparams_names, args, ovr=args.ovr, binary=True)
        res = gp_minimize(train.train, models[model][1], n_calls=args.n_calls, random_state=42)

        features_cutoff = None
        param_grid = {}
        best_params = res['x']
        for name, param in zip(hparams_names, best_params):
            if name == 'features_cutoff':
                features_cutoff = param
            if name == 'threshold':
                threshold = param
            elif name != 'features_cutoff':
                param_grid[name] = param
        try:
            assert features_cutoff is not None
        except AttributeError:
            exit('features_cutoff not in the hyperparameters. Leaving')

        if args.scaler != 'binarize':
            scaler = get_scaler(args.scaler)()
        else:
            from sklearn.preprocessing import Binarizer

            scaler = Pipeline([('minmax', get_scaler('minmax')()), ('binarizer', Binarizer(threshold=threshold))])
        if features_cutoff > data["test"]["data"].shape[1]:
            features_cutoff = data["test"]["data"].shape[1]
        if args.correct_batches:
            df_train = pd.DataFrame(data["train"]["data"][:, :features_cutoff], index=train_classes)
            df_test = pd.DataFrame(data["test"]["data"][:, :features_cutoff], index=test_classes)
            df = pd.concat((df_train, df_test), 0).T
            all_batches = np.concatenate((train_batches, test_batches))
            if low_test_data is not None:
                df_low_test = pd.DataFrame(data["lows"]["data"][:, :features_cutoff], index=low_test_classes)
                df = pd.concat((df, df_low_test.T), 1)
                all_batches = np.concatenate((all_batches, low_test_batches))
            all_train = pycombat(df, all_batches).T.values
            all_x_train = all_train[:data["train"]["data"].shape[0]]
            test_data = all_train[
                        data["train"]["data"].shape[0]:(data["train"]["data"].shape[0] + data["test"]["data"].shape[0])]
            if low_test_data is not None:
                low_test_data = all_train[(data["train"]["data"].shape[0] + data["test"]["data"].shape[0]):]

            del all_train
            # x_valid = pycombat(pd.DataFrame(x_valid, index=y_valid).T, batches_valid).T.values
        else:
            all_x_train = data["train"]["data"][:, :features_cutoff]
            test_data = data["test"]["data"][:, :features_cutoff]
            low_test_data = data["lows"]["data"][:, :features_cutoff]

        m = models[model][0]()
        m.set_params(**param_grid)

        all_x_train = scaler.fit_transform(all_x_train)
        try:
            test_data = scaler.transform(test_data)
        except:
            exit('Unresolved problem!')
        m.fit(all_x_train, train_classes)
        test_score = m.score(test_data, test_classes)

        train_score = m.score(all_x_train, train_classes)
        y_preds_test = m.predict(test_data)
        y_preds_train = m.predict(all_x_train)
        mcc_test = MCC(test_classes, y_preds_test)
        mcc_train = MCC(train_classes, y_preds_train)
        try:
            y_proba_train = m.predict_proba(all_x_train)
            y_proba_test = m.predict_proba(test_data)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), train_classes)])
            y_top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                   zip(y_proba_test.argsort(1), test_classes)])
        except:
            y_top3_valid = -1
            y_top3_test = -1
        if train.best_scores_train is not None:
            param_grid = {}
            for name, param in zip(hparams_names, best_params):
                param_grid[name] = param
            best_params_dict[model] = param_grid

            best_params_dict[model]['train_acc_mean'] = np.mean(train.best_scores_train)
            best_params_dict[model]['train_acc_std'] = np.std(train.best_scores_train)
            best_params_dict[model]['valid_acc_mean'] = np.mean(train.best_scores_valid)
            best_params_dict[model]['valid_acc_std'] = np.std(train.best_scores_valid)
            best_params_dict[model]['test_acc'] = test_score

            best_params_dict[model]['train_mcc_mean'] = np.mean(train.best_mccs_train)
            best_params_dict[model]['train_mcc_std'] = np.std(train.best_mccs_train)
            best_params_dict[model]['valid_mcc_mean'] = np.mean(train.best_mccs_valid)
            best_params_dict[model]['valid_mcc_std'] = np.std(train.best_mccs_valid)
            best_params_dict[model]['test_mcc'] = mcc_test

            best_params_dict[model]['train_top3_mean'] = np.mean(train.best_top3_train)
            best_params_dict[model]['train_top3_std'] = np.std(train.best_top3_train)
            best_params_dict[model]['valid_top3_mean'] = np.mean(train.best_top3_valid)
            best_params_dict[model]['valid_top3_std'] = np.std(train.best_top3_valid)
            best_params_dict[model]['test_top3'] = y_top3_test

        if model not in previous_models:
            previous_models[model]['valid_acc_mean'] = -1
        if low_test_data is not None:
            low_test_data = scaler.transform(low_test_data)
            low_test_score = m.score(low_test_data, low_test_classes)
            y_preds_lows = m.predict(low_test_data)
            mcc_lows = MCC(low_test_classes, y_preds_lows)
            print(f'test score: {test_score}, low test score: {low_test_score}')
            if float(best_params_dict[model]['valid_acc_mean']) >= float(previous_models[model]['valid_acc_mean']):
                best_params_dict[model]['test_lows_acc'] = low_test_score
                best_params_dict[model]['test_lows_mcc'] = mcc_lows
                fig = get_confusion_matrix(y_preds_lows, low_test_classes, unique_labels)
                save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_lows_test",
                                      acc=low_test_score)
                try:
                    best_params_dict[model]['test_auc'] = save_roc_curve(m, low_test_data, low_test_classes,
                                                                         unique_labels,
                                                                         f"{args.destination}/ROC/{model}_lows_test",
                                                                         binary=args.binary, acc=low_test_score)
                except:
                    print('No proba function, or something else.')

        else:
            print(f'test score: {test_score}')
        if float(best_params_dict[model]['valid_acc_mean']) >= float(previous_models[model]['valid_acc_mean']):
            model_filename = f"{args.destination}/saved_models/sklearn/{model}.sav"
            with open(model_filename, 'wb') as file:
                pickle.dump(m, file)
            scaler_filename = f"{args.destination}/saved_models/sklearn/scaler_{model}.sav"
            with open(scaler_filename, 'wb') as file:
                pickle.dump(scaler, file)

            fig = get_confusion_matrix(y_preds_test, test_classes, unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_test", acc=test_score)
            try:
                save_roc_curve(m, test_data, test_classes, unique_labels, f"{args.destination}/ROC/{model}_test",
                               binary=args.binary,
                               acc=test_score)
            except:
                print('No proba function, or something else.')

    for name in best_params_dict.keys():
        if name in previous_models.keys():
            prev_valid_acc = float(previous_models[name]['valid_acc_mean'])
        else:
            prev_valid_acc = -1
        if float(best_params_dict[name]['valid_acc_mean']) > prev_valid_acc:
            for param in best_params_dict[name].keys():
                best_params_dict[name][param] = str(best_params_dict[name][param])
        else:
            for param in previous_models[name].keys():
                best_params_dict[name][param] = str(previous_models[name][param])
    for name in previous_models.keys():
        if name not in best_params_dict.keys():
            best_params_dict[name] = {}
            for param in previous_models[name].keys():
                best_params_dict[name][param] = str(previous_models[name][param])

    with open(f'{args.destination}/saved_models/sklearn/best_params.json', "w") as read_file:
        json.dump(best_params_dict, read_file)
