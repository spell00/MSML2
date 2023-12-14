#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHUL of Québec, Canada
Author: Simon Pelletier
August 2022

"""


import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_paths(path, paths):
    for x in os.listdir(path):
        new_path = f"{path}/{x}"
        if 'cm' in os.listdir(new_path):
            paths += [new_path]
            return paths
        else:
            get_paths(new_path, paths)
    return paths


if __name__ == "__main__":

    results = pd.DataFrame()
    paths_list = []
    root = f"logs/ae_classifier/old_data/"
    paths_list = get_paths(root, paths_list)
    values = {}
    params = ['zinb']

    for path in paths_list:
        arg = ''
        for param in params:
            val = [x for x in path.split('/') if param in x][0]
            if len(arg) == 0:
                arg = val
            else:
                arg = f"{arg}/{val}"
        if arg not in values:
            values[arg] = {}
        values[arg][path] = {}

    best_values_paths = {}
    for arg in list(values.keys()):
        best_path = ''
        best_closs = np.inf
        best_event_acc = None
        for path in values[arg]:
            event_acc = EventAccumulator(f"{path}/traces")
            event_acc.Reload()
            try:
                closs = event_acc.Tensors('valid/loss')
            except:
                continue
            closs = tf.make_ndarray(closs[0].tensor_proto).item()
            if closs < best_closs:
                best_closs = closs
                best_path = path
                best_event_acc = event_acc

        best_values_paths[arg] = {}
        # event_acc = EventAccumulator(f"{best_path}/hp")
        # event_acc.Reload()
        if best_event_acc is not None:
            for name in best_event_acc.summary_metadata.keys():
                if name not in ['_hparams_/experiment', '_hparams_/session_start_info']:
                    value = best_event_acc.Tensors(name)
                    value = tf.make_ndarray(value[0].tensor_proto).item()
                    if np.isnan(value):
                        value = 'nan'
                    best_values_paths[arg][name] = value
            # results.to_csv('summary_table_tb.csv', index=None)

    with open(f"results/best_{'_'.join(params)}.json", "w") as outfile:
        json.dump(best_values_paths, outfile)
        # json_object = json.dumps(best_values_paths)
        # json.dump(json_object, outfile)

