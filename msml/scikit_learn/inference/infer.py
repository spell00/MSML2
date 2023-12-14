#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from sklearn.metrics import matthews_corrcoef as mcc
from msml.scikit_learn.utils import load_data


def infer(classifier, scale, data):
    """
    Function used to infer from a trained model

    """
    unique_labels = []
    for label in train_labels:
        if label not in unique_labels:
            unique_labels += [label]
    unique_labels = np.array(unique_labels)
    data['train']['classes'] = np.array(
        [np.argwhere(label == unique_labels)[0][0] for label in data['train']['labels']]
    )
    data['tests']['classes'] = np.array(
        [np.argwhere(label == unique_labels)[0][0] for label in data['tests']['labels']]
    )
    data['lows']['classes'] = np.array(
        [np.argwhere(label == unique_labels)[0][0] for label in data['low']['labels']]
    )

    with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r",
              encoding="utf-8") as json_file:
        features_cutoff = int(json.load(json_file)['RandomForestClassifier']['features_cutoff'])
    if args.correct_batches:
        data["train"]["data"] = pd.DataFrame(
            data["train"]["data"][:, :features_cutoff], index=data['train']['classes']
        )
        data["tests"]["data"] = pd.DataFrame(
            data["tests"]["data"][:, :features_cutoff], index=data['tests']['classes']
        )
        data["all"]["data"] = pd.concat(
            (
                data["train"]["data"],
                data["tests"]["data"]
            ), 0).T
        data["all"]["batches"] = np.concatenate(
                (data["train"]["batches"], data["tests"]["batches"])
        )

        data["lows"]["tests"] = pd.DataFrame(
            data["lows"]["data"][:, :features_cutoff], index=data['lows']['classes']
        )
        data["all"]["data"] = pd.concat((data["all"]["data"], data["lows"]["tests"].T), 1)
        data["all"]["batches"] = np.concatenate((data["all"]["batches"], data['lows']['batches']))
        data["all"]["data"] = pycombat(data["all"]["data"], data["all"]["batches"]).T.values

        len_train = data["train"]["data"].shape[0]
        len_train_test = (len_train + data["tests"]["data"].shape[0])

        data["train"]["data"] = data["all"]["data"][:data["train"]["data"].shape[0]]
        data["tests"]["data"] = data["all"]["data"][len_train:len_train_test]
        data["lows"]["data"] = data["all"]["data"][len_train:len_train_test]
    else:
        data["train"]["data"] = data["train"]["data"][:, :features_cutoff]
        data["tests"]["data"] = data["tests"]["data"][:, :features_cutoff]

    data["train"]["data"] = scale.transform(data["train"]["data"][:, :features_cutoff])
    data["tests"]["data"] = scale.transform(data["tests"]["data"][:, :features_cutoff])
    data["lows"]["data"] = scale.transform(data["lows"]["data"][:, :features_cutoff])

    data["train"]["score"] = classifier.score(data["train"]["data"], data["train"]["classes"])
    data["train"]["mcc"] = mcc(data["train"]["classes"], classifier.predict(data["train"]["data"]))

    data["tests"]["score"] = classifier.score(data["tests"]["data"], data["tests"]["classes"])
    data["tests"]["mcc"] = mcc(data["tests"]["classes"], classifier.predict(data["tests"]["data"]))

    data["lows"]["score"] = classifier.score(data["lows"]["data"],
                                             classifier.predict(data["lows"]["data"]))
    data["lows"]["mcc"] = mcc(data["lows"]["classes"], classifier.predict(data["lows"]["data"]))

    print("train ACC", data["train"]["score"], 'tests ACC:', data["tests"]["score"],
          'lows ACC:', data["lows"]["score"])
    print("train MCC", data["train"]["mcc"], 'tests MCC:', data["tests"]["mcc"],
          'lows MCC:', data["lows"]["mcc"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--on', type=str, default="all")
    parser.add_argument('--ovr', type=int, default=0, help='OneVsAll strategy')
    parser.add_argument('--remove_zeros_on', type=str, default="lows")
    parser.add_argument('--drop_lows', type=str, default="train")
    parser.add_argument('--drop_blks', type=int, default=0)
    parser.add_argument('--correct_batches', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--run_name', type=str, default='1')
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--scaler', type=str, default='minmax')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--jackknife', type=str, default="False")
    parser.add_argument("--input_dir", type=str, default='src1/resources/matrices/',
                        help="Path to intensities csv file")
    parser.add_argument("--output_dir", type=str, default='src1/results/',
                        help="Path to intensities csv file")
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
    args.inputs_destination = f"{args.input_dir}/on{args.on}/" \
                              f"remove_zeros_on{args.remove_zeros_on}/{args.run_name}/"
    args.destination = f"{args.output_dir}/on{args.on}/" \
                       f"remove_zeros_on{args.remove_zeros_on}/{args.scaler}/" \
                       f"corrected{args.correct_batches}/drop_lows{args.drop_lows}/" \
                       f"drop_blks{args.drop_blks}/" \
                       f"binary{args.binary}/boot{args.jackknife}/{args.scaler}/" \
                       f"cv{args.n_splits}/nrep{args.n_repeats}/ovr{args.ovr}/{args.run_name}/"

    args.train_input_binary = f"{args.inputs_destination}/ALL_train_inputs_gt0.15.csv"
    args.test_input_binary = f"{args.inputs_destination}/ALL_test_inputs_gt0.15.csv"
    args.train_input = f"{args.inputs_destination}/ALL_train_inputs_gt0.15.csv"
    args.test_input = f"{args.inputs_destination}/ALL_test_inputs_gt0.15.csv"

    with open(f"{args.destination}/saved_models/sklearn/RandomForestClassifier.sav",
              'rb', encoding='utf-8') as pickle_file:
        model = pickle.load(pickle_file)
    with open(f"{args.destination}/saved_models/sklearn/RandomForestClassifier.sav",
              'rb', encoding='utf-8') as pickle_file:
        scaler = pickle.load(pickle_file)

    train_labels = []
    train_data = []
    low_test_data = []
    low_test_labels = []
    low_test_batches = []
    low_train_data = []
    low_train_labels = []
    low_train_batches = []
    train_lows = []
    train_batches = []
    test_labels = []
    test_data = []
    test_batches = []
    test_lows = []
    if args.drop_lows == "train":
        train = load_data(args.train_input, drop_l=False, drop_b=args.drop_blks,
                          binary=args.binary)
        test = load_data(args.test_input, drop_l=False, drop_b=args.drop_blks,
                         binary=args.binary)

        test_low_inds = np.arange(len(test['lows']))[test['lows'] == 1]
        train_low_inds = np.arange(len(train['lows']))[train['lows'] == 1]

        low_train_data = train['data'][train_low_inds]
        low_train_labels = train['labels'][train_low_inds]
        low_train_batches = train['batches'][train_low_inds]

        low_test_data = test['data'][test_low_inds]
        low_test_labels = test['labels'][test_low_inds]
        low_test_batches = test['batches'][test_low_inds]

        test_data = np.delete(test['data'], test_low_inds, axis=0)
        test_labels = np.delete(test['labels'], test_low_inds, axis=0)
        test_batches = np.delete(test['batches'], test_low_inds, axis=0)
        train_data = np.delete(train['data'], train_low_inds, axis=0)
        train_labels = np.delete(train['labels'], train_low_inds, axis=0)
        train_batches = np.delete(train['batches'], train_low_inds, axis=0)

        low_test_data = np.concatenate((low_train_data, low_test_data))
        low_test_labels = np.concatenate((low_test_labels, low_train_labels))
        low_test_batches = np.concatenate((low_test_batches, low_train_batches))

    elif args.drop_lows == "tests":
        train = load_data(args.train_input, drop_l=False, drop_b=args.drop_blks,
                          binary=args.binary)
        test = load_data(args.test_input, drop_l=True, drop_b=args.drop_blks,
                         binary=args.binary)

        low_inds = np.arange(len(test['lows']))[test['lows'] == 1]
        train_data = np.concatenate((train['data'], test['data'][low_inds]))
        train_labels = np.concatenate((train['labels'], test['labels'][low_inds]))
        train_batches = np.concatenate((train['batches'], test['batches'][low_inds]))

        test_data = np.delete(test['data'], low_inds, axis=0)
        test_labels = np.delete(test['labels'], low_inds, axis=0)
        test_batches = np.delete(test['batches'], low_inds, axis=0)

    elif args.drop_lows == "all":
        train = load_data(args.train_input, drop_l=True,
                          drop_b=args.drop_blks, binary=args.binary)
        test = load_data(args.test_input, drop_l=True,
                         drop_b=args.drop_blks, binary=args.binary)
    else:
        sys.exit("drop_lows must be one of: train, tests or all")

    data_dict = {
        'train': {
            'labels': train_labels,
            'data': train_data,
            'batches': train_batches,
            'lows': train_lows,
        },
        'tests': {
            'labels': test_labels,
            'data': test_data,
            'batches': test_batches,
            'lows': test_lows,
        },
    }
    if low_test_data is not None:
        data_dict['lows'] = {
            'labels': low_test_labels,
            'data': low_test_data,
            'batches': low_test_batches,
        }
    elif low_train_data is not None:
        data_dict['lows'] = {
            'labels': low_train_labels,
            'data': low_train_data,
            'batches': low_train_batches,
        }
    else:
        data_dict['lows'] = {
            'labels': None,
            'data': None,
            'batches': None,
        }

    infer(model, scaler, data_dict)
