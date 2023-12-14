#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import json
import sys
import warnings
import numpy as np
import matplotlib
from skopt import gp_minimize
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import matthews_corrcoef as mcc
from msml.scikit_learn.train.params_gp import models as models_gp
from msml.scikit_learn.train.params import models as models_grid
from msml.scikit_learn.train.utils import get_variables
from msml.scikit_learn.utils import get_scaler, get_confusion_matrix, save_confusion_matrix, save_roc_curve
from msml.utils.utils import get_unique_labels
from msml.scikit_learn.train.sklearn_train import Train

matplotlib.use('Agg')

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src1/models/sklearn/'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--optim', type=str, default='gp')
    parser.add_argument('--n_calls', type=int, default=10)
    parser.add_argument('--on', type=str, default="all")
    parser.add_argument('--ovr', type=int, default=0, help='OneVsAll strategy')
    parser.add_argument('--remove_zeros_on', type=str, default="all")
    parser.add_argument('--drop_lows', type=str, default="train")
    parser.add_argument('--drop_blks', type=int, default=0)
    parser.add_argument('--correct_batches', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--run_name', type=str, default='1')
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--n_splits', type=int, default=2)
    parser.add_argument('--scaler', type=str, default='minmax')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--jackknife', type=str, default="False")
    parser.add_argument("--input_dir", type=str, default='resources/matrices/',
                        help="Path to intensities csv file")
    parser.add_argument("--output_dir", type=str, default='results/',
                        help="Path to intensities csv file")
    parser.add_argument("--mz_bin", type=str, default="1")
    parser.add_argument("--rt_bin", type=str, default="1")
    args = parser.parse_args()

    if args.verbose == 0:
        args.verbose = False
    else:
        args.verbose = True

    if args.ovr == 0:
        args.ovr = False
    else:
        args.ovr = True

    if args.drop_blks == 0:
        args.drop_blks = False
    else:
        args.drop_blks = True

    if args.binary == 0:
        args.binary = False
    else:
        args.binary = True

    if args.correct_batches == 0:
        args.correct_batches = False
    else:
        args.correct_batches = True

    if args.random == 0:
        args.random = False
    else:
        args.random = True

    if args.jackknife == "True":
        args.jackknife = True
    else:
        args.jackknife = False
    args.inputs_destination = f"{args.input_dir}/mz{args.mz_bin}/rt{args.rt_bin}/on{args.on}/" \
                              f"remove_zeros_on{args.remove_zeros_on}/" \
                              f"binary{args.binary}/{args.run_name}/"
    args.destination = f"{args.output_dir}/mz{args.mz_bin}/rt{args.rt_bin}/on{args.on}/remove_zeros_on{args.remove_zeros_on}" \
                       f"/{args.scaler}/corrected{args.correct_batches}/" \
                       f"drop_lows{args.drop_lows}/drop_blks{args.drop_blks}/" \
                       f"binary{args.binary}/boot{args.jackknife}/{args.scaler}/" \
                       f"cv{args.n_splits}/nrep{args.n_repeats}/ovr{args.ovr}/" \
                       f"{args.run_name}/"

    args.train_input_binary = f"{args.inputs_destination}/BACT_train_inputs_gt0.15.csv"
    args.test_input_binary = f"{args.inputs_destination}/BACT_test_inputs_gt0.15.csv"
    args.train_input = f"{args.inputs_destination}/BACT_train_inputs_gt0.15.csv"
    args.test_input = f"{args.inputs_destination}/BACT_test_inputs_gt0.15.csv"

    if "best_params.json" in f'{args.destination}/saved_models/sklearn':
        with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r",
                  encoding="utf-8") as json_file:
            previous_models = json.load(json_file)
    else:
        print("No previous best parameters found.")
        previous_models = {}
    best_params_dict = previous_models
    os.makedirs(f"{args.destination}/saved_models/sklearn/", exist_ok=True)
    if args.optim == 'gp':
        models = models_gp
    elif args.optim == 'grid':
        models = models_grid

    print('Getting the data...')
    variables = get_variables(args)
    for model_name, (model, param_grid) in models.items():
        print(f"Training {model_name}")
        if args.optim == 'gp':
            hparams_names = [x.name for x in param_grid]
        elif args.optim == 'grid':
            hparams_names = list(param_grid.keys())
        # TODO At this step, some concentrations might be omitted for training. Now it always
        # trains on highs
        train = Train([model_name, model], hparams_names, args, variables)
        if args.optim == 'gp':
            res = gp_minimize(train.train, param_grid, n_calls=args.n_calls, random_state=42)
            best_params = res['x']

        elif args.optim == 'grid':
            BEST_ACC = -1
            for features_cutoff in [1, 10, 100, 1000, 10000]:
                for h_param in ParameterGrid(param_grid):
                    acc = train.train(list(h_param.values()))
                    if acc > BEST_ACC:
                        best_params = h_param
                        BEST_ACC = acc
        else:
            sys.exit(f"Optim {args.optim} is invalid.")
        FEATURES_CUTOFF = 0
        param_grid = {}
        for name, param in zip(hparams_names, best_params):
            if name == 'features_cutoff':
                FEATURES_CUTOFF = param
            else:
                param_grid[name] = param
        try:
            assert FEATURES_CUTOFF is not None
        except AttributeError:
            sys.exit('features_cutoff not in the hyperparameters. Leaving')

        data = train.get_data()
        scaler = get_scaler(args.scaler)()

        classifier = models[model_name][0]()
        classifier.set_params(**param_grid)

        # TODO At this step, scores for all separate conditions are required
        all_x_train = scaler.fit_transform(data['train']["highs"]['data'][:, :FEATURES_CUTOFF])
        test_data = scaler.transform(data['tests']["highs"]['data'][:, :FEATURES_CUTOFF])

        classifier.fit(all_x_train, data['train']["highs"]['classes'])
        test_score = classifier.score(test_data, data['tests']["highs"]['classes'])

        train_score = classifier.score(all_x_train, data['train']["highs"]['classes'])
        y_preds_test = classifier.predict(test_data)
        y_preds_train = classifier.predict(all_x_train)
        mcc_test = mcc(data['tests']["highs"]['classes'], y_preds_test)
        mcc_train = mcc(data['train']["highs"]['classes'], y_preds_train)
        y_proba_train = classifier.predict_proba(all_x_train)
        y_proba_test = classifier.predict_proba(test_data)
        y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                     zip(y_proba_train.argsort(1), data['train']["highs"]['classes'])])
        y_top3_test = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                     zip(y_proba_test.argsort(1), data['tests']["highs"]['classes'])])
        best = train.get_best_scores()
        if best['valid']['score'] is not None:
            param_grid = {}
            for name, param in zip(hparams_names, best_params):
                param_grid[name] = param
            best_params_dict[model_name] = param_grid

            best_params_dict[model_name]['train_acc_mean'] = np.mean(best['train']['score'])
            best_params_dict[model_name]['train_acc_std'] = np.std(best['train']['score'])
            best_params_dict[model_name]['valid_acc_mean'] = np.mean(best['valid']['score'])
            best_params_dict[model_name]['valid_acc_std'] = np.std(best['valid']['score'])
            best_params_dict[model_name]['test_acc'] = test_score

            best_params_dict[model_name]['train_mcc_mean'] = np.mean(best['train']['mcc'])
            best_params_dict[model_name]['train_mcc_std'] = np.std(best['train']['mcc'])
            best_params_dict[model_name]['valid_mcc_mean'] = np.mean(best['valid']['mcc'])
            best_params_dict[model_name]['valid_mcc_std'] = np.std(best['valid']['mcc'])
            best_params_dict[model_name]['test_mcc'] = mcc_test

            best_params_dict[model_name]['train_top3_mean'] = np.mean(best['train']['top3'])
            best_params_dict[model_name]['train_top3_std'] = np.std(best['train']['top3'])
            best_params_dict[model_name]['valid_top3_mean'] = np.mean(best['valid']['top3'])
            best_params_dict[model_name]['valid_top3_std'] = np.std(best['valid']['top3'])
            best_params_dict[model_name]['test_top3'] = y_top3_test
        if model not in previous_models:
            previous_models[model_name]['valid_acc_mean'] = -1
        if len(data['tests']['lows'].keys()) != 0:
            data['tests']['lows']['data'] = scaler.transform(data['tests']['lows']['data'][:, :FEATURES_CUTOFF])
            low_test_score = classifier.score(data['tests']['lows']['data'],
                                              data['tests']['lows']['classes'])
            y_preds_lows = classifier.predict(data['tests']['lows']['data'])
            mcc_lows = mcc(data['tests']['lows']['classes'], y_preds_lows)
            print(f'tests score: {test_score}, low tests score: {low_test_score}')
            if float(best_params_dict[model_name]['valid_acc_mean']) >= \
                float(previous_models[model_name]['valid_acc_mean']):
                best_params_dict[model_name]['test_lows_acc'] = low_test_score
                best_params_dict[model_name]['test_lows_mcc'] = mcc_lows
                unique_labels = get_unique_labels(
                    data['tests'][list(data['tests'].keys())[0]]['labels']
                )
                conf_matrix = get_confusion_matrix(y_preds_lows, data['tests']['lows']['classes'],
                                                   unique_labels)
                save_confusion_matrix(conf_matrix, f"{args.destination}/confusion_matrices/"
                                                   f"{model_name}_lows_test", acc=low_test_score)
                try:
                    # Replace the new dict by just data["tests"]["lows"], but first must change
                    # the names for "data" and "targets"
                    save_roc_curve(classifier,
                                   {'data': data['tests']['lows']['data'],
                                    'targets': data['tests']['lows']['classes']},
                                   unique_labels, f"{args.destination}/ROC/{model_name}_lows_test",
                                   acc=low_test_score)
                except IOError as e:
                    print(f'{e}\nNo proba function, or something else.')

        else:
            print(f'tests score: {test_score}')

    for name in best_params_dict.keys():
        if name in previous_models.keys():
            PREV_VALID_ACC = float(previous_models[name]['valid_acc_mean'])
        else:
            PREV_VALID_ACC = -1
        if float(best_params_dict[name]['valid_acc_mean']) > PREV_VALID_ACC:
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

    with open(f'{args.destination}/saved_models/sklearn/best_params.json', "w", encoding="utf-8")\
            as read_file:
        json.dump(best_params_dict, read_file)
