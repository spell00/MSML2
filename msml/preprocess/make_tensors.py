#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on previous work by Elsa CLAUDE - CHUL of Québec, Canada
February 2020 - August 2020
Modified by Simon Pelletier
June 2021

"""

from datetime import datetime

start_time = datetime.now()
import os
import warnings
import logging
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PIL import Image
from matplotlib import cm
from pickle import dump
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from msml.utils.features_selection import get_feature_selection_method, keep_only_not_zeros, keep_not_zeros, \
    process_data, count_array, make_lists, split_df, MultiKeepNotFunctions

warnings.filterwarnings("ignore")
if 'logs' not in os.listdir():
    os.makedirs('logs')


class MakeTensorsMultiprocess:
    """
    Concat
    """

    def __init__(self, tsv_list, labels_list, bins, test_run, n_samples, log, path, save):
        """
        :param tsv_list:
        :param labels_list:
        :param test_run:
        :return:
        """
        os.makedirs(f'{path}/images', exist_ok=True)
        os.makedirs(f'{path}/csv', exist_ok=True)

        self.bins = bins

        self.path = path
        self.tsv_list = tsv_list
        self.labels_list = labels_list
        self.save = save
        self.test_run = test_run
        self.log2 = log
        if n_samples != -1:
            self.tsv_list = self.tsv_list[:n_samples]
            self.labels_list = self.labels_list[:n_samples]

    def process(self, i):
        """

        This process makes a 'tensor' (list of dataframes)
        :param i:
        :return:
        """
        file, label = self.tsv_list[i], self.labels_list[i]
        try:
            tsv = pd.read_csv(file, header=0, sep='\t', dtype=np.float64)
        except:
            exit('Error reading csv')
        print(f"Processing file {i}: {file} min_parents: min={tsv.min_parent_mz.min()} max={tsv.min_parent_mz.max()}")

        tsv = tsv[tsv.bin_intensity != 0]

        tsv = tsv.drop(['max_parent_mz'], axis=1)
        tsv['mz_bin'] = tsv['mz_bin'].round(2)

        min_parents_mz = np.unique(tsv.min_parent_mz)
        interval = min_parents_mz[1] - min_parents_mz[0]
        if self.bins['mz_shift']:
            mz_shift = self.bins['mz_bin_post'] / 2
        else:
            mz_shift = 0
        if self.bins['rt_shift']:
            rt_shift = self.bins['rt_bin_post'] / 2
        else:
            rt_shift = 0

        final = {min_parent: pd.DataFrame(
            np.zeros([int(np.ceil(tsv.rt_bin.max() / self.bins['rt_bin_post'])) + 1,
                      int(np.ceil(tsv.mz_bin.max() / self.bins['mz_bin_post'])) + 1]),
            dtype=np.float64,
            index=np.arange(0, tsv.rt_bin.max() + self.bins['rt_bin_post'], self.bins['rt_bin_post']).round(
                self.bins['rt_rounding']) - rt_shift,
            columns=np.arange(0, tsv.mz_bin.max() + self.bins['mz_bin_post'], self.bins['mz_bin_post']).round(
                self.bins['mz_rounding']) - mz_shift
        ) for min_parent in np.arange(int(tsv.min_parent_mz.min()), int(tsv.min_parent_mz.max()) + interval, interval)}
        for i in list(final.keys()):
            final[i].index = np.round(final[i].index, self.bins['rt_rounding'])
            final[i].columns = np.round(final[i].columns, self.bins['mz_rounding'])

        for i, line in enumerate(tsv.values):
            min_parent, rt, mz, intensity = line
            if np.isnan(rt) or np.isnan(mz) or np.isnan(min_parent):
                continue
            if self.bins['rt_shift']:
                tmp_rt = np.floor(np.round(rt / self.bins['rt_bin_post'], 8)) * self.bins['rt_bin_post']
                if rt % (self.bins['rt_bin_post'] / 2) > (self.bins['rt_bin_post'] / 2):
                    tmp_rt += self.bins['rt_bin_post'] / 2
                else:
                    tmp_rt -= self.bins['rt_bin_post'] / 2
                rt = tmp_rt
            else:
                rt = np.floor(np.round(rt / self.bins['rt_bin_post'], 8)) * self.bins['rt_bin_post']

            if self.bins['mz_shift']:
                tmp_mz = np.floor(np.round(mz / self.bins['mz_bin_post'], 8)) * self.bins['mz_bin_post']
                if mz % (self.bins['mz_bin_post'] / 2) > (self.bins['mz_bin_post'] / 2):
                    tmp_mz += self.bins['mz_bin_post'] / 2
                else:
                    tmp_mz -= self.bins['mz_bin_post'] / 2
                mz = tmp_mz
            else:
                mz = np.floor(np.round(mz / self.bins['mz_bin_post'], 8)) * self.bins['mz_bin_post']
            if self.bins['rt_rounding'] != 0:
                rt = np.round(rt, self.bins['rt_rounding'])
            if self.bins['mz_rounding'] != 0:
                mz = np.round(mz, self.bins['mz_rounding'])
            if self.log2 == 'inloop':
                final[min_parent][mz][rt] += np.log1p(intensity)
            else:
                final[min_parent][mz][rt] += intensity
            if self.test_run and i == 10:
                break
        os.makedirs(f"{self.path}/nibabel/", exist_ok=True)
        img = nib.Nifti1Image(np.stack(list(final.values())), np.eye(4))
        nib.save(img, f'{self.path}/nibabel/{label}.nii')
        if label == '20220825_blk_p11_03':
            print('stop')
        if self.save:
            _ = [self.save_images_and_csv(matrix, f"{label}", min_parent) for i, (min_parent, matrix) in
                 enumerate(zip(final, list(final.values())))]

        # return np.stack(list(final.values())), list(final.keys()), label

    def n_samples(self):
        """
        Gets n samples. Mainly just to have a second class so pylint does not complain.
        """
        return len(self.tsv_list)

    def save_images_and_csv(self, final, label, i):
        os.makedirs(f"{self.path}/csv3d/{label}/", exist_ok=True)
        os.makedirs(f"{self.path}/images3d/{label}/", exist_ok=True)
        final.to_csv(f"{self.path}/csv3d/{label}/{label}_{i}.csv", index_label='ID')
        im = Image.fromarray(np.uint8(cm.gist_earth(final.values) * 255))
        im.save(f"{self.path}/images3d/{label}/{label}_{i}.png")
        im.close()


def make_df(dirinput, dirname, bins, args_dict, names_to_keep=None, features=None):
    """
    Concatenate and transpose

    :param dirinput:
    :param dirname:
    :param args_dict:
    :return:
    """
    # Get list of all wanted tsv

    # path = f"{dirname}/"
    lists = make_lists(dirinput, dirname, args_dict.run_name)
    if names_to_keep is not None:
        inds_to_keep = [i for i, x in enumerate(lists['labels']) if
                        x.split('_')[1] in names_to_keep]
        lists["tsv"] = np.array(lists['tsv'])[inds_to_keep].tolist()
        lists["labels"] = np.array(lists['labels'])[inds_to_keep].tolist()
    concat = MakeTensorsMultiprocess(lists["tsv"], lists["labels"], bins,
                                     args_dict.test_run, args_dict.n_samples, args_dict.log2, dirname, args_dict.save)

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

    pool.map(concat.process,
             range(len(concat.tsv_list))
             )
    print('Tensors are done.')


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--test_run", type=int, default=0,
                        help="Is it a test run? 1 for true 0 for false")
    parser.add_argument("--n_samples", type=int, default=-1,
                        help="How many samples to run? Only modify to make test runs faster (-1 for all samples)")
    parser.add_argument("--log2", type=str, default='inloop',
                        help='log the data in the loop, after the loop or not log the data. Choices: [inloop, after, no]')
    parser.add_argument("--shift", type=int, default=0, help='Shift the data matrix')
    parser.add_argument("--binary", type=int, default=0, help='Blanks vs bacteria')
    parser.add_argument("--threshold", type=float, default=0.1)
    # parser.add_argument("--mz_rounding", type=int, default=1)
    # parser.add_argument("--rt_rounding", type=int, default=1)
    parser.add_argument("--ms_level", type=int, default=2)
    parser.add_argument("--mz_bin_post", type=float, default=0.2)
    parser.add_argument("--rt_bin_post", type=float, default=20)
    parser.add_argument("--mz_bin", type=float, default=0.01)
    parser.add_argument("--rt_bin", type=float, default=0.1)
    parser.add_argument("--run_name", type=str, default="eco,sag,efa,kpn,blk,pool")
    parser.add_argument("--scaler", type=str, default="none")
    parser.add_argument("--combat_corr", type=int, default=0)
    parser.add_argument("--use_test", type=int, default=0)
    parser.add_argument("--use_valid", type=int, default=0)
    parser.add_argument("--k", type=str, default=-1, help="Number of features to keep")
    parser.add_argument("--save", type=int, default=1, help="Save images and csvs?")
    parser.add_argument("--resources_path", type=str, default='../../../resources',
                        help="Path to input directory")
    parser.add_argument("--experiment", type=str, default='new_old_data')
    parser.add_argument("--feature_selection", type=str, default='mutual_info_classif',
                        help="Mutual Information classification cutoff")
    parser.add_argument("--feature_selection_threshold", type=float, default=0.,
                        help="Mutual Information classification cutoff")
    parser.add_argument("--spd", type=str, default="200")
    parser.add_argument('--batch_removal_method', type=str, default='none')
    args = parser.parse_args()
    args.combat_corr = 0  # TODO to remove

    if float(args.mz_bin) >= 1:
        args.mz_bin = int(float(args.mz_bin))
    else:
        args.mz_bin = float(args.mz_bin)
    if float(args.rt_bin) >= 1:
        args.rt_bin = int(float(args.rt_bin))
    else:
        args.rt_bin = float(args.rt_bin)
    if float(args.mz_bin_post) >= 1:
        args.mz_bin_post = int(float(args.mz_bin_post))
    else:
        args.mz_bin_post = float(args.mz_bin_post)
    if float(args.rt_bin_post) >= 1:
        args.rt_bin_post = int(float(args.rt_bin_post))
    else:
        args.rt_bin_post = float(args.rt_bin_post)

    if args.mz_bin_post < 1:
        args.mz_rounding = len(str(args.mz_bin_post).split('.')[-1]) + 1
    else:
        args.mz_rounding = 1

    if args.rt_bin_post < 1:
        args.rt_rounding = len(str(args.rt_bin_post).split('.')[-1]) + 1
    else:
        args.rt_rounding = 1

    bins = {
        'mz_bin_post': args.mz_bin_post,
        'rt_bin_post': args.rt_bin_post,
        'mz_rounding': args.mz_rounding,
        'rt_rounding': args.rt_rounding,
        'mz_shift': args.shift,
        'rt_shift': args.shift
    }

    out_dest = f"{args.resources_path}/{args.experiment}/matrices"
    input_dir = f"{args.resources_path}/{args.experiment}/tsv"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    dir_name = f"{script_dir}/{out_dest}/mz{args.mz_bin}/rt{args.rt_bin}/mzp{args.mz_bin_post}/" \
               f"rtp{args.rt_bin_post}/{args.spd}spd/ms{args.ms_level}/combat{args.combat_corr}/" \
               f"shift{args.shift}/{args.scaler}/log{args.log2}/{args.feature_selection}/"
    dir_input = f"{script_dir}/{input_dir}/mz{args.mz_bin}/rt{args.rt_bin}/{args.spd}spd/ms{args.ms_level}/all/"

    bacteria_to_keep = None

    if len(args.run_name.split(',')) > 1:
        bacteria_to_keep = args.run_name.split(',')
    else:
        print('No plate specified, using them all')
        bacteria_to_keep = None

    print('dir_input: ', dir_input, dir_name)

    return args, dir_input, dir_name, bins, bacteria_to_keep


if __name__ == "__main__":
    args, dir_input, dir_name, bins, bacteria_to_keep = parse_arguments()
    logging.basicConfig(filename=f'logs/make_tensors_ms{args.ms_level}_split.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    make_df(dir_input, dir_name, bins=bins, args_dict=args,
            names_to_keep=bacteria_to_keep)
