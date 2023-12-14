# TODO This script cannot do cross validation. This is for when the data is already split. Each sample in train and
# TODO valid must be from seperate batch. If training and validating on the same batches, a bias is introduced

# !/usr/bin/env python3
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
import re
import csv
import json
import matplotlib

matplotlib.use('Agg')
from sklearn.model_selection import StratifiedKFold
from msml.scikit_learn.utils import get_scaler, get_confusion_matrix, save_confusion_matrix, save_roc_curve
from sklearn.metrics import matthews_corrcoef as MCC
from skopt import gp_minimize
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from msml.scikit_learn.train.params_gp import *
from msml.utils.utils import get_unique_labels
from msml.utils.batch_effect_removal import remove_batch_effect, get_berm

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src/models/sklearn/'


class Train:
    def __init__(self, model, hparams_names, args, ovr, binary=True):
        self.args = args
        self.best_roc_score = -1
        self.berm = get_berm(args.batch_removal_method)
        try:
            with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
                self.previous_models = {}
        except:
            self.previous_models = {}
        self.ovr = ovr
        self.binary = binary
        self.model = model
        self.hparams_names = hparams_names

        self.scores = {}
        self.best_scores = {}
        for group in ['train', 'valid']:
            self.scores[group] = {}
            self.best_scores[group] = {}
            for metric in ['mcc', 'acc', 'top3']:
                self.scores[group][metric] = []
                self.best_scores[group][metric] = []
        self.y_preds = np.array([])
        self.y_valids = np.array([])
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

        if args.scaler != 'binarize':
            scaler = get_scaler(args.scaler)()
        else:
            from sklearn.preprocessing import Binarizer
            scaler = Pipeline([('minmax', get_scaler('minmax')()), ('binarizer', Binarizer(threshold=threshold))])

        print(f'Iteration: {self.iter}')

        y_pred_valids = []
        y_valids = []
        x_valids = []

        data = remove_batch_effect(self.berm, self.data['inputs']['all'], self.data['inputs']['train'],
                                   self.data['inputs']['valid'], self.data['inputs']['test'],
                                   self.data['batches']['all'])
        x_train = scaler.fit_transform(data['train'].iloc[:, :features_cutoff])
        x_valid = scaler.transform(data['valid'].iloc[:, :features_cutoff])

        m = self.model()
        m.set_params(**param_grid)
        if self.ovr:
            m = OneVsRestClassifier(m)
        try:
            m.fit(x_train, self.data['cats']['train'])
        except:
            return 1

        score_valid = m.score(x_valid, self.data['cats']['valid'])
        score_train = m.score(x_train, self.data['cats']['train'])
        print('valid_score:', score_valid, "features_cutoff", features_cutoff, 'h_params:', param_grid)
        self.scores['train']['acc'] = score_train
        self.scores['valid']['acc'] = score_valid

        y_pred_train = m.predict(x_train)
        y_pred_valid = m.predict(x_valid)

        y_pred_valids.extend(y_pred_valid)
        y_valids.extend(self.data['cats']['valid'])
        x_valids.extend(x_valid)

        self.scores['train']['mcc'] = MCC(self.data['cats']['train'], y_pred_train)
        self.scores['valid']['mcc'] = MCC(self.data['cats']['valid'], y_pred_valid)
        subs = list(self.data['subs']['all'].keys())
        try:
            y_proba_train = m.predict_proba(x_train)
            y_proba_valid = m.predict_proba(x_valid)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), data['cats']['train'])])
            y_top3_valid = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_valid.argsort(1), data['cats']['valid'])])

            self.scores['train']['top3'] = y_top3_train
            self.scores['valid']['top3'] = y_top3_valid
        except:
            # not all models can make predictions. -1 means the model could not make a prediction
            self.scores['train']['top3'] = -1
            self.scores['valid']['top3'] = -1

        if self.best_scores['valid']['acc'] is None:
            self.best_scores['valid']['acc'] = 0

        if self.best_scores['valid']['acc'] == []:
            self.best_scores['valid']['acc'] = 0

        # score = np.mean(score_valid)
        if np.isnan(score_valid):
            score_valid = 0
        if model not in self.previous_models:
            self.previous_models[model] = {}
            self.previous_models[model]['valid'] = {}
            self.previous_models[model]['valid']['acc'] = -1

        if score_valid > self.best_scores['valid']['acc'] and score_valid > float(self.previous_models[model]['valid']['acc']):
            for group in ['train', 'valid']:
                for metric in ['acc', 'mcc', 'top3']:
                    self.best_scores[group][metric] = self.scores[group][metric]

            fig = get_confusion_matrix(y_valids, y_pred_valids, self.unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid", acc=score_valid)
            # fig = get_confusion_matrix(y_valids_highs, y_pred_valids_highs, unique_labels)
            # save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid_highs",
            #                       acc=np.mean(self.best_scores_valid_highs))
            # fig = get_confusion_matrix(y_valids_lows, y_pred_valids_lows, unique_labels)
            # save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid_lows",
            #                       acc=np.mean(self.best_scores_valid_lows))
            try:
                self.best_roc_score = save_roc_curve(m, x_valids, y_valids, self.unique_labels,
                                                     f"{args.destination}/ROC/{model}_valid", binary=args.binary,
                                                     acc=score_valid)
            except:
                pass
            """
            try:
                self.best_roc_score_highs = save_roc_curve(m, x_valids_highs, y_valids_highs, unique_labels,
                                                           f"{args.destination}/ROC/{model}_valid_highs",
                                                           binary=args.binary,
                                                           acc=np.mean(self.best_scores_valid_highs))
            except:
                pass
            try:
                self.best_roc_score_lows = save_roc_curve(m, x_valids_lows, y_valids_lows, unique_labels,
                                                          f"{args.destination}/ROC/{model}_valid_lows",
                                                          binary=args.binary,
                                                          acc=np.mean(self.best_scores_valid_lows))
            except:
                pass
            """
        return 1 - score_valid

    def get_data(self, csvs_path):
        """


        Returns: Nothing

        """
        unique_labels = []
        data = {}
        for info in ['subs', 'inputs', 'names', 'labels', 'cats', 'batches']:
            data[info] = {}
            for group in ['all', 'all_pool', 'train', 'train_pool', 'valid', 'valid_pool', 'test', 'test_pool']:
                data[info][group] = np.array([])
        for group in ['train', 'test', 'valid']:
            if group == 'valid' and not self.args.use_valid:
                skf = StratifiedKFold(n_splits=5)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train']).__next__()
                data['inputs']['train'], data['inputs']['valid'] = data['inputs']['train'].iloc[train_inds], \
                                                                   data['inputs']['train'].iloc[valid_inds]
                data['labels']['train'], data['labels']['valid'] = data['labels']['train'][train_inds], \
                                                                   data['labels']['train'][valid_inds]
                data['names']['train'], data['names']['valid'] = data['names']['train'].iloc[train_inds], \
                                                                 data['names']['train'].iloc[valid_inds]
                data['batches']['train'], data['batches']['valid'] = data['batches']['train'][train_inds], \
                                                                     data['batches']['train'][valid_inds]
                data['cats']['train'], data['cats']['valid'] = data['cats']['train'][train_inds], data['cats']['train'][
                    valid_inds]
                subcategories = np.unique(
                    ['v' for x in data['names'][group]])
                subcategories = np.array([x for x in subcategories if x != ''])
                data['subs'][group] = {x: np.array([]) for x in subcategories}
                for sub in list(data['subs'][group]):
                    data['subs']['train'][sub], data['subs']['valid'][sub] = data['subs']['train'][sub][train_inds], \
                                                                             data['subs']['train'][sub][valid_inds]

            if group == 'test' and not self.args.use_test:
                skf = StratifiedKFold(n_splits=5)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train']).__next__()
                data['inputs']['train'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                                                                  data['inputs']['train'].iloc[test_inds]
                data['names']['train'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                                                                data['names']['train'].iloc[test_inds]
                data['labels']['train'], data['labels']['test'] = data['labels']['train'][train_inds], \
                                                                  data['labels']['train'][test_inds]
                data['batches']['train'], data['batches']['test'] = data['batches']['train'][train_inds], \
                                                                    data['batches']['train'][test_inds]
                data['cats']['train'], data['cats']['test'] = \
                    data['cats']['train'][train_inds], data['cats']['train'][test_inds]
                subcategories = np.unique(
                    ['v' for x in data['names'][group]])
                subcategories = np.array([x for x in subcategories if x != ''])
                data['subs'][group] = {x: np.array([]) for x in subcategories}
                for sub in list(data['subs'][group]):
                    data['subs']['train'][sub], data['subs']['test'][sub] = data['subs']['train'][sub][train_inds], \
                                                                            data['subs']['train'][sub][test_inds]

            else:
                data['inputs'][group] = pd.read_csv(
                    f"{csvs_path}/{group}_inputs.csv"
                )
                data['names'][group] = data['inputs'][group]['ID']
                data['labels'][group] = np.array([d.split('_')[1] for d in data['names'][group]])
                unique_labels = get_unique_labels(data['labels'][group])
                # data['batches'][group] = np.array([int(d.split('_')[0]) for d in data['names'][group]])
                # try:
                data['batches'][group] = np.array([int(''.join(re.split('\D+', d.split('_')[2]))) for d in data['names'][group]])
                # except:
                #     data['batches'][group] = np.array([int(re.split('\w+', d.split('_')[2])) for d in data['names'][group]])

                # Drops the ID column
                data['inputs'][group] = data['inputs'][group].iloc[:, 1:]
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])
                subcategories = np.unique(
                    [re.split('\d+', x.split('_')[3])[0] for x in data['names'][group]])
                subcategories = np.array([x for x in subcategories if x != ''])
                data['subs'][group] = {x: np.array([]) for x in subcategories}
                for sub in subcategories:
                    data['subs'][group][sub] = np.array(
                        [i for i, x in
                         enumerate(data['names'][group]) if sub in x.split('_')[3]]
                    )

                # TODO this should not be necessary
                if group != 'train':
                    data['inputs'][group] = data['inputs'][group].loc[:, data['inputs']['train'].columns]

        subcategories = np.unique(
            np.concatenate(
                [np.unique(
                    [re.split('\d+', x.split('_')[3])[0] for x in data['names'][group]]) for group in
                    list(data['names'].keys())
                ]
            )
        )
        subcategories = np.array([x for x in subcategories if x != ''])
        self.subcategories = subcategories

        for key in list(data.keys()):
            if key == 'inputs':
                data[key]['all'] = pd.concat((data[key]['train'], data[key]['valid'], data[key]['test']), 0)
            elif key != 'subs':
                data[key]['all'] = np.concatenate((data[key]['train'], data[key]['valid'], data[key]['test']), 0)

        # Add values for sets without a subset
        for s in subcategories:
            for group in ['train', 'valid', 'test']:
                if s not in list(data['subs'][group].keys()):
                    data['subs'][group][s] = np.array([-1 for _ in range(len(data['cats'][group]))])

        data['subs']['all'] = {sub: np.concatenate(
            (data['subs']['train'][sub], data['subs']['valid'][sub], data['subs']['test'][sub]), 0) for sub in
            subcategories}
        unique_batches = np.unique(data['batches']['all'])
        for group in ['train', 'valid', 'test', 'all']:
            data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

        self.data = data
        self.unique_labels = unique_labels
        self.unique_batches = unique_batches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--transductive', type=float, default=0, help='not functional')
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--n_calls', type=int, default=20)
    parser.add_argument('--log2', type=str, default='inloop', help='inloop or after')
    parser.add_argument('--ovr', type=int, default=0, help='OneVsAll strategy')
    # parser.add_argument('--drop_lows', type=str, default="none")
    # parser.add_argument('--drop_blks', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--feature_selection', type=str, default='mutual_info_classif',
                        help='mutual_info_classif or f_classif')
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--spd', type=int, default=200)
    parser.add_argument('--mz', type=float, default=0.2)
    parser.add_argument('--rt', type=float, default=20.0)
    parser.add_argument('--min_rt', type=float, default=0)
    parser.add_argument('--min_mz', type=float, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--run_name', type=str, default='all')
    parser.add_argument('--scaler', type=str, default='robust')
    parser.add_argument('--preprocess_scaler', type=str, default='none')
    # parser.add_argument('--n_repeats', type=int, default=1)
    # parser.add_argument('--jackknife', type=int, default=0)
    parser.add_argument('--inference_inputs', type=int, default=0)
    parser.add_argument("--input_dir", type=str, help="Path to intensities csv file")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to intensities csv file")
    parser.add_argument("--combat", type=int, default=0)
    parser.add_argument("--combat_data", type=int, default=0)
    parser.add_argument("--shift", type=str, default=0)
    parser.add_argument('--batch_removal_method', type=str, default='none')
    parser.add_argument('--use_valid', type=int, default=1, help='Use if valid data is in a seperate file')
    parser.add_argument('--use_test', type=int, default=1, help='Use if test data is in a seperate file')
    args = parser.parse_args()
    if args.combat == 0:
        args.combat_data = 0

    args.destination = f"{args.output_dir}/mz{args.mz}/rt{args.rt}/minmz{args.min_mz}/minrt{args.min_rt}/" \
                       f"{args.spd}spd/{args.scaler}/combat{args.combat}_{args.combat_data}/shift{args.shift}/" \
                       f"{args.preprocess_scaler}/log{args.log2}/corrected{args.batch_removal_method}/" \
                       f"binary{args.binary}/{args.scaler}/cv{args.n_splits}/" \
                       f"ovr{args.ovr}/thres{args.threshold}/{args.run_name}/inference{args.inference_inputs}"

    csvs_path = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                f"shift{args.shift}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/{args.run_name}/"
    if args.combat_data and args.combat:
        train_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                      f"shift{args.shift}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/{args.run_name}/" \
                      f"train_inputs_gt0.0_combat.csv"
    else:
        train_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                      f"shift{args.shift}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/{args.run_name}/" \
                      f"train_inputs.csv"
    valid_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                  f"shift{args.shift}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/{args.run_name}/" \
                  f"valid_inputs.csv"
    test_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                 f"shift{args.shift}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/{args.run_name}/" \
                 f"test_inputs.csv"

    models = {
        # "BaggingClassifier": [BaggingClassifier, bag_space],
        "LinearSVC": [LinearSVC, linsvc_space],
        "KNeighbors": [KNeighborsClassifier, kn_space],
        "SVCRBF": [SVC, svcrbf_space],
        "NuSVC": [NuSVC, nusvc_space],
        "RandomForestClassifier": [RandomForestClassifier, rfc_space],
        # "LogisticRegression": [LogisticRegression, logreg_space],
        "Gaussian_Naive_Bayes": [GaussianNB, nb_space],
        # "QDA": [QuadraticDiscriminantAnalysis, qda_space],
        "SGDClassifier": [SGDClassifier, sgd_space],
        "SVCLinear": [SVC, svc_space],
        # "LDA": [LinearDiscriminantAnalysis, lda_space],  # Creates an error...
        # "AdaBoost_Classifier": [AdaBoostClassifier, param_grid_ada],
        # "Voting_Classifier": [VotingClassifier, param_grid_voting],
    }

    try:
        with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
            previous_models = json.load(json_file)
    except:
        previous_models = {}
    best_params_dict = previous_models
    os.makedirs(f"{args.destination}/saved_models/sklearn/", exist_ok=True)
    for model in models:
        print(f"Training {model}")
        hparams_names = [x.name for x in models[model][1]]
        train = Train(models[model][0], hparams_names, args, ovr=args.ovr, binary=True)
        train.get_data(csvs_path)
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
        if features_cutoff > train.data['inputs']["test"].shape[1]:
            features_cutoff = train.data['inputs']["test"].shape[1]

        train_data = train.data["inputs"]["train"].iloc[:, :features_cutoff]
        valid_data = train.data["inputs"]["valid"].iloc[:, :features_cutoff]
        test_data = train.data["inputs"]["test"].iloc[:, :features_cutoff]

        m = models[model][0]()
        m.set_params(**param_grid)

        transductive = 0
        if transductive:
            scaler.fit(np.concatenate((train_data,
                                       valid_data,
                                       test_data), 0))
        else:
            scaler.fit(train_data)
        try:
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)
        except:
            exit('Unresolved problem!')
        m.fit(train_data, train.data["cats"]["train"])
        test_score = m.score(test_data, train.data["cats"]["test"])
        train_score = m.score(train_data, train.data["cats"]["train"])
        y_preds_test = m.predict(test_data)
        y_preds_train = m.predict(train_data)
        mcc_test = MCC(train.data["cats"]["test"], y_preds_test)
        mcc_train = MCC(train.data["cats"]["train"], y_preds_train)
        try:
            y_proba_train = m.predict_proba(train_data)
            y_proba_test = m.predict_proba(test_data)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), train.data["cats"]["train"])])
            y_top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                   zip(y_proba_test.argsort(1), train.data["cats"]["test"])])
        except:
            y_top3_valid = -1
            y_top3_test = -1
        if train.best_scores['train']['acc'] is not None:
            param_grid = {}
            for name, param in zip(hparams_names, best_params):
                param_grid[name] = param
            best_params_dict[model] = param_grid
            for group in ['train', 'valid']:
                best_params_dict[model][group] = {}
                for metric in ['acc', 'mcc', 'top3']:
                    best_params_dict[model][group][metric] = train.best_scores[group][metric]
                    for k in list(train.data['subs']['all'].keys()):
                        inds = torch.concat(lists[group]['concs'][k]).detach().cpu().numpy()
                        inds = np.array([i for i, x in enumerate(inds) if x > -1])
                        if len(inds) > 0:
                            traces[group][f'mcc_{k}'] = MCC(preds[inds], classes[inds])
                        else:
                            traces[group][f'mcc_{k}'] = -1

            best_params_dict[model]['test'] = {}
            best_params_dict[model]['test']['acc'] = test_score
            best_params_dict[model]['test']['mcc'] = mcc_test
            best_params_dict[model]['test']['top3'] = y_top3_test

        if model not in previous_models:
            try:
                previous_models[model]['valid']['acc'] = -1
            except:
                pass
        else:
            print(f'test score: {test_score}')

        if float(best_params_dict[model]['valid']['acc']) >= float(previous_models[model]['valid']['acc']):
            model_filename = f"{args.destination}/saved_models/sklearn/{model}.sav"
            with open(model_filename, 'wb') as file:
                pickle.dump(m, file)
            scaler_filename = f"{args.destination}/saved_models/sklearn/scaler_{model}.sav"
            with open(scaler_filename, 'wb') as file:
                pickle.dump(scaler, file)

            fig = get_confusion_matrix(train.data["cats"]["test"], y_preds_test, train.unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_test", acc=test_score)
            try:
                save_roc_curve(m, test_data, train.data["cats"]["test"], train.unique_labels,
                               f"{args.destination}/ROC/{model}_test",
                               binary=args.binary, acc=test_score)
            except:
                print('No proba function, or something else.')

    for name in best_params_dict.keys():
        if name in previous_models.keys():
            prev_valid_acc = float(previous_models[name]['valid']['acc'])
        else:
            prev_valid_acc = -1
        if float(best_params_dict[name]['valid']['acc']) > prev_valid_acc:
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
