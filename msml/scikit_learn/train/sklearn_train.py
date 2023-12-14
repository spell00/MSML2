#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
from combat.pycombat import pycombat
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold,\
    RepeatedKFold, StratifiedShuffleSplit

from msml.scikit_learn.utils import get_scaler, get_confusion_matrix, \
    save_confusion_matrix, save_roc_curve
from msml.utils.utils import get_unique_labels

matplotlib.use('Agg')
np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src1/models/sklearn/'


class Train:
    """
    Train class
    """

    def __init__(self, train_model, hparams_names, args, variables):
        """

        :param train_model:
        :param hparams_names:
        :param args:
        """
        try:
            with open(
                f'{args.destination}/saved_models/sklearn/best_params.json', "r", encoding='utf-8'
            ) as j_file:
                self.previous_models = json.load(j_file)
        except IOError:
            self.previous_models = {}
        self.args = args
        self.model = train_model
        self.hparams_names = hparams_names
        # self.train_indices, self.test_indices, _ = split_train_test(self.labels)
        self.iter = 0

        self.scores = {
            'current': {
                "train": {
                    'score': [],
                    'top3': [],
                    'mcc': [],
                    'targets': np.array([])
                },

                "valid": {
                    'score': [],
                    'mcc': [],
                    'top3': [],
                    'targets': np.array([])
                },
            },
            'best': {
                "train": {
                    'score': [-1],
                    'mcc': [-1],
                    'top3': [-1],
                    'targets': np.array([])
                },

                "valid": {
                    'score': [-1],
                    'top3': [-1],
                    'mcc': [-1],
                    'targets': np.array([])
                },
            }
        }
        self.variables = variables

    def get_data(self):
        """
        Gets data

        :return:
        """
        return self.variables

    def get_best_scores(self):
        """
        Gets best scores

        :return:
        """
        return self.scores['best']

    def correct_batches(self):
        """
        Corrects data for batches

        :return:
        """
        df_train = pd.DataFrame(
            self.variables['train']['highs']['data'][:, :10000],
            index=self.variables['train']['highs']['classes']
        )
        df_test = pd.DataFrame(self.variables['tests']['highs']['data'][:, :10000],
                               index=self.variables['tests']['highs']['classes'])
        df_all = pd.concat((df_train, df_test), 0).T
        all_batches = np.concatenate((
            self.variables['train']['highs']['batches'],
            self.variables['tests']['highs']['batches']
        ))
        if self.variables["tests"]["lows"] is not None:
            df_low_test = pd.DataFrame(self.variables["tests"]["lows"]["data"][:, :10000],
                                       index=self.variables["tests"]["lows"]["classes"])
            df_all = pd.concat((df_all, df_low_test.T), 1)
            all_batches = np.concatenate((
                all_batches, self.variables["tests"]["lows"]["batches"]
            ))
        self.variables['train']['highs']['data'] = pycombat(
            df_all,
            all_batches
        ).T.values[:len(self.variables["train"]['highs']["data"])]

    def train(self, h_params):
        """
        Train function. Must be a class with only one parameter other than self in order
        to use it for a gaussian process optimization
        :param h_params:
        :return:
        """

        self.iter += 1
        feats_cutoff = None
        param_grid = {}
        for param_name, param in zip(self.hparams_names, h_params):
            if param_name == 'features_cutoff':
                feats_cutoff = param
            else:
                param_grid[param_name] = param
        assert feats_cutoff is not None
        skf = self.get_skf()

        # This needs to be modified if not using just highs for training
        scaler = get_scaler(self.args.scaler)()
        if self.args.correct_batches:
            self.correct_batches()
        print(f'Iteration: {self.iter}')

        collector = {
            'predictions': [],
            'targets': [],
            'data': [],
        }
        train_nums = np.arange(0, len(self.variables['train']['highs']['data']))
        for i, (train_inds, valid_inds) in enumerate(
            skf.split(train_nums, self.variables['train']['highs']['classes'])
        ):
            # Just plot the first iteration, it will already be crowded if doing > 100
            print(f"CV: {i + 1}")
            x_train = self.variables['train']['highs']['data'][train_inds][:, :feats_cutoff]
            y_train = self.variables['train']['highs']['classes'][train_inds]
            x_valid = self.variables['train']['highs']['data'][valid_inds][:, :feats_cutoff]
            y_valid = self.variables['train']['highs']['classes'][valid_inds]

            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)
            model_name, model = self.model
            model = model()
            model.set_params(**param_grid)
            if self.args.ovr:
                model = OneVsRestClassifier(model)
            try:
                model.fit(x_train, y_train)
            except BrokenPipeError:
                print(BrokenPipeError)

            self.scores['current']['train']['score'] += [model.score(x_train, y_train)]
            self.scores['current']['valid']['score'] += [model.score(x_valid, y_valid)]
            preds = model.predict(x_valid)
            collector['predictions'].extend(preds)
            collector['targets'].extend(y_valid)
            collector['data'].extend(x_valid)
            print('valid_score:', self.scores['current']['valid']['score'][-1],
                  "features_cutoff", feats_cutoff, 'h_params:', param_grid)
            self.scores['current']['train']['mcc'] += [mcc(y_train, model.predict(x_train))]
            self.scores['current']['valid']['mcc'] += [mcc(y_valid, preds)]
            y_proba_train = model.predict_proba(x_train)
            y_proba_test = model.predict_proba(x_valid)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), y_train)])
            y_top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                   zip(y_proba_test.argsort(1), y_valid)])
            self.scores['current']['train']['top3'] += [y_top3_train]
            self.scores['current']['valid']['top3'] += [y_top3_test]

        if self.scores['best']['valid']['score'] is None:
            self.scores['best']['valid']['score'] = np.array([0])
        self.evaluate_scores(model, model_name, collector)
        return 1 - np.mean(self.scores['best']['valid']['score'])

    def train_steps(self):
        """
        Trains on binary, then multiclass

        :return:
        """
        return 1 - np.mean(self.scores['best']['valid']['score'])

    def get_skf(self):
        """
        Gets split strategy

        :return:
        """
        if self.args.n_splits != -1:
            if self.args.n_repeats > 1:
                if self.args.jackknife:
                    self.args.jackknife = False
                    self.args.n_repeats = 1
                    print('Jacknifing cannot be combined with Repeated Holdout. '
                          'Doing Repeated Holdout.')
                skf = RepeatedStratifiedKFold(n_repeats=self.args.n_repeats,
                                              n_splits=self.args.n_splits)
            elif self.args.jackknife:
                skf = StratifiedShuffleSplit(n_splits=self.args.n_splits, test_size=0.2)
            else:
                skf = StratifiedKFold(n_splits=self.args.n_splits, shuffle=True, random_state=1)
        else:
            if self.args.jackknife:
                print('Jacknifing cannot be combined with LOOCV. Doing Repeated LOOCV')
                self.args.jackknife = False
            if self.args.n_repeats > 1:
                skf = RepeatedKFold(n_repeats=self.args.n_repeats,
                                    n_splits=len(self.variables['train']['classes']))
            else:
                skf = KFold(n_splits=len(self.variables['train']['classes']), shuffle=True,
                            random_state=1)
        return skf

    def evaluate_scores(self, model, model_name, collector):
        """
        Records best scores

        :param model:
        :param model_name:
        :param collector:
        :return:
        """
        score = np.mean(self.scores['current']['valid']['score'])
        if model_name not in self.previous_models:
            self.previous_models[model_name] = {'valid_acc_mean': -1}
        if score > np.mean(self.scores['best']['valid']['score']) \
            and score > float(self.previous_models[model_name]['valid_acc_mean']):

            self.scores['best']['train']['score'] = self.scores['current']['train']['score']
            self.scores['best']['valid']['score'] = self.scores['current']['valid']['score']
            self.scores['best']['train']['mcc'] = self.scores['current']['train']['mcc']
            self.scores['best']['valid']['mcc'] = self.scores['current']['valid']['mcc']
            self.scores['best']['train']['top3'] = self.scores['current']['train']['top3']
            self.scores['best']['valid']['top3'] = self.scores['current']['valid']['top3']
            unique_labels = get_unique_labels(self.variables['train']["highs"]['labels'])
            conf_matrix = get_confusion_matrix(collector['predictions'],
                                               collector['targets'], unique_labels)
            save_confusion_matrix(
                conf_matrix,
                f"{self.args.destination}/confusion_matrices/{model_name}_valid",
                acc=score
            )
            try:
                save_roc_curve(model, collector, unique_labels,
                               f"{self.args.destination}/ROC/{model_name}_valid", acc=score)
            except IOError:
                pass
