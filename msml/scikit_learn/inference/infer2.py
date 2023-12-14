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
from msml.scikit_learn.utils import load_data, load_data2


def infer(classifier, scale, data):
    """
    Function used to infer from a trained model

    """
    unique_labels = []
    for label in train['labels']:
        if label not in unique_labels:
            unique_labels += [label]
    unique_labels = np.array(unique_labels)
    data['train']['classes'] = np.array(
        [np.argwhere(label == unique_labels)[0][0] for label in data['train']['labels']]
    )
    data['tests']['classes'] = np.array(
        [np.argwhere(label == unique_labels)[0][0] for label in data['tests']['labels']]
    )
    # data['lows']['classes'] = np.array(
    #     [np.argwhere(label == unique_labels)[0][0] for label in data['low']['labels']]
    # )

    with open(f'results/mz{args.mz_bin}/rt{args.rt_bin}/minmz{args.min_mz}/minrt{args.min_rt}/{args.spd}spd/{args.scaler}/corrected{args.correct_batches}/drop_lows{args.drop_lows}/drop_blks0/binary0/boot0/{args.scaler}/cv5/nrep1/ovr0/1/saved_models/sklearn/best_params.json', "r",
              encoding="utf-8") as json_file:
        features_cutoff = int(json.load(json_file)[args.model]['features_cutoff'])
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

        # data["lows"]["tests"] = pd.DataFrame(
        #     data["lows"]["data"][:, :features_cutoff], index=data['lows']['classes']
        # )
        # data["all"]["data"] = pd.concat((data["all"]["data"], data["lows"]["tests"].T), 1)
        # data["all"]["batches"] = np.concatenate((data["all"]["batches"], data['lows']['batches']))
        data["all"]["data"] = pycombat(data["all"]["data"], data["all"]["batches"]).T.values

        len_train = data["train"]["data"].shape[0]
        len_train_test = (len_train + data["tests"]["data"].shape[0])

        data["train"]["data"] = data["all"]["data"][:data["train"]["data"].shape[0]]
        data["tests"]["data"] = data["all"]["data"][len_train:len_train_test]
        # data["lows"]["data"] = data["all"]["data"][len_train:len_train_test]
    else:
        data["train"]["data"] = data["train"]["data"][:, :features_cutoff]
        data["tests"]["data"] = data["tests"]["data"][:, :features_cutoff]

    data["train"]["data"] = scale.transform(data["train"]["data"][:, :features_cutoff])
    data["tests"]["data"] = scale.transform(data["tests"]["data"][:, :features_cutoff])
    # data["lows"]["data"] = scale.transform(data["lows"]["data"][:, :features_cutoff])

    data["train"]["score"] = classifier.score(data["train"]["data"], data["train"]["classes"])
    data["train"]["mcc"] = mcc(data["train"]["classes"], classifier.predict(data["train"]["data"]))

    data["tests"]["score"] = classifier.score(data["tests"]["data"], data["tests"]["classes"])
    data["tests"]["mcc"] = mcc(data["tests"]["classes"], classifier.predict(data["tests"]["data"]))

    # data["lows"]["score"] = classifier.score(data["lows"]["data"],
    #                                          classifier.predict(data["lows"]["data"]))
    # data["lows"]["mcc"] = mcc(data["lows"]["classes"], classifier.predict(data["lows"]["data"]))

    print("train ACC", data["train"]["score"], 'tests ACC:', data["tests"]["score"])
          # 'lows ACC:', data["lows"]["score"])
    print("train MCC", data["train"]["mcc"], 'tests MCC:', data["tests"]["mcc"])
          # 'lows MCC:', data["lows"]["mcc"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--on', type=str, default="all")
    parser.add_argument('--ovr', type=int, default=0, help='OneVsAll strategy')
    parser.add_argument('--remove_zeros_on', type=str, default="lows")
    parser.add_argument('--drop_lows', type=str, default="no")
    parser.add_argument('--run_name', type=str, default="14032022-231234")
    parser.add_argument('--drop_blks', type=int, default=0)
    parser.add_argument('--correct_batches', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--scaler', type=str, default='binarize')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--jackknife', type=str, default="False")
    parser.add_argument("--input_dir", type=str, default='resources/matrices/',
                        help="Path to intensities csv file")
    parser.add_argument('--min_rt', type=float, default=0.0)
    parser.add_argument('--min_mz', type=float, default=0.0)
    parser.add_argument('--spd', type=float, default=200)
    parser.add_argument('--model', type=str, default='LinearSVC')
    parser.add_argument('--rt_bin', type=float, default=10)
    parser.add_argument('--mz_bin', type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default='results/',
                        help="Path to intensities csv file")
    args = parser.parse_args()

    args.inputs_destination = f"resources/matrices/mz{args.mz_bin}/rt{args.rt_bin}/{args.spd}spd/bNone/" \
                              f"{args.run_name}/"
    args.inference_destination = f"resources/new_data/matrices/mz0.01/rt10/200spd/bNone/{args.run_name}/"

    args.destination = f"{args.output_dir}/mz{args.mz_bin}/rt{args.rt_bin}/minmz{args.min_mz}/minrt{args.min_rt}/" \
                       f"{args.spd}spd/binarize/corrected{args.correct_batches}/" \
                       f"drop_lowsno/drop_blks0/binary0/boot0/binarize/cv5/nrep1/ovr0/1/"
    args.train_input_binary = f"{args.inputs_destination}/BACT_train_inputs_gt0.15.csv"
    args.test_input_binary = f"{args.inference_destination}/BACT_inference_inputs_{args.run_name}.csv"
    args.train_input = f"{args.inputs_destination}/BACT_train_inputs_gt0.15.csv"
    args.test_input = f"{args.inference_destination}/BACT_inference_inputs_{args.run_name}.csv"

    with open(f"results/mz{args.mz_bin}/rt{args.rt_bin}/minmz{args.min_mz}/minrt{args.min_rt}/{args.spd}spd/"
              f"{args.scaler}/corrected{args.correct_batches}/drop_lowsno/drop_blks0/binary0/boot0/"
              f"{args.scaler}/cv5/nrep1/ovr0/1/saved_models/sklearn/{args.model}.sav", 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    with open(f"results/mz{args.mz_bin}/rt{args.rt_bin}/minmz{args.min_mz}/minrt{args.min_rt}/{args.spd}spd/"
              f"{args.scaler}/corrected{args.correct_batches}/drop_lowsno/drop_blks0/binary0/"
              f"boot0/{args.scaler}/cv5/nrep1/ovr0/1/saved_models/sklearn/scaler_{args.model}.sav",
              'rb') as pickle_file:
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

    train = load_data(args.train_input, drop_l=False,
                      drop_b=args.drop_blks, binary=args.binary, min_rt=args.min_rt, min_mz=args.min_mz)
    test = load_data2(args.test_input, drop_l=False,
                      drop_b=args.drop_blks, binary=args.binary, min_rt=args.min_rt, min_mz=args.min_mz)

    data_dict = {
        'train': {
            'labels': train['labels'],
            'data': train['data'],
            'batches': train['batches'],
            'lows': train['lows'],
        },
        'tests': {
            'labels': test['labels'],
            'data': test['data'],
            'batches': test['batches'],
            'lows': test['lows'],
        },
        'all': {
            'labels': np.array([]),
            'data': pd.DataFrame(),
            'batches': np.array([]),
            'lows': np.array([]),
        }
    }

    infer(model, scaler, data_dict)
