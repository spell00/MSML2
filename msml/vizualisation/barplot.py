#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri September 28 2021
@author: Simon Pelletier
"""

import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

if __name__ == "__main__":
    NAME = 'saved_models/sklearn/best_params_binaryFalse_300_cv5_nrep1_bootFalse_dropLowsTrue.json'
    with open(NAME, encoding="utf-8") as f:
        best_params_steps_binaryFalse = json.load(f)
    valid_means_binaryFalse = []
    valid_stds_binaryFalse = []
    for i, results in enumerate(best_params_steps_binaryFalse):
        valid_means_binaryFalse += [float(best_params_steps_binaryFalse[results]["valid_acc_mean"])]
        valid_stds_binaryFalse += [float(best_params_steps_binaryFalse[results]["valid_acc_std"])]
    test_values_binaryFalse = []
    for i, results in enumerate(best_params_steps_binaryFalse):
        test_values_binaryFalse += [float(best_params_steps_binaryFalse[results]["test_acc"])]
    NAME = 'saved_models/sklearn/best_params_binaryTrue_300_cv5_nrep1_dropLowsTrue.json'
    with open(NAME, encoding='utf-8') as f:
        best_params_steps_binaryTrue = json.load(f)
    valid_means_binaryTrue = []
    valid_stds_binaryTrue = []
    for i, results in enumerate(best_params_steps_binaryTrue):
        valid_means_binaryTrue += [float(best_params_steps_binaryTrue[results]["valid_acc_mean"])]
        valid_stds_binaryTrue += [float(best_params_steps_binaryTrue[results]["valid_acc_std"])]
    test_values_binaryTrue = []
    for i, results in enumerate(best_params_steps_binaryTrue):
        test_values_binaryTrue += [float(best_params_steps_binaryTrue[results]["test_acc"])]

    plt.figure(figsize=(15, 10))
    ind = np.arange(len(best_params_steps_binaryFalse))
    bar_valid_binaryTrue = plt.bar(best_params_steps_binaryTrue.keys(), valid_means_binaryTrue,
                                   width=0.18, label='Valid acc 300 spd (binary)')
    plt.errorbar(ind, valid_means_binaryTrue, 0, valid_stds_binaryTrue, barsabove=True, fmt='none')

    bar_test_binaryTrue = plt.bar(ind + 0.18, test_values_binaryTrue, width=0.18,
                                  label='Test acc 300 spd (binary)')

    bar_valid_binaryFalse = plt.bar(ind + 0.42, valid_means_binaryFalse, width=0.18,
                                    label='Valid acc 300 spd')
    plt.errorbar(ind + 0.42, valid_means_binaryFalse, 0, valid_stds_binaryFalse, barsabove=True,
                 fmt='none')
    plt.xticks(ind + 0.28, best_params_steps_binaryFalse.keys(), rotation='vertical')

    bar_test_binaryFalse = plt.bar(ind + 0.6, test_values_binaryFalse, width=0.2,
                                   label='Test acc 300 spd')
    min300 = min(min(valid_means_binaryFalse), min(test_values_binaryFalse))
    min200 = min(min(valid_means_binaryTrue), min(test_values_binaryTrue))

    plt.ylim(min(min200, min300) - 0.05)

    plt.legend(handles=[bar_valid_binaryTrue, bar_test_binaryTrue, bar_valid_binaryFalse,
                        bar_test_binaryFalse, ], loc='upper right')
    plt.tight_layout()
    plt.savefig('saved_models/sklearn/models_results_best_params_300spd_nolows.png')
    plt.close()
