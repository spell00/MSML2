#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import sys

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msml.scikit_learn.utils import split_train_test
import re
from matplotlib import rcParams, cycler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.spatial.distance import cdist

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    FROM https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def rand_jitter(arr):
    """
    Returns the arr plus random gaussian noise

    :param arr:
    :return:
    """
    return arr + np.random.randn(len(arr)) * 0.01


def pca(data, labels_str, name, ord_name):
    """
    Plots a PCA

    :param data:
    :param labels_str:
    :param name:
    :param ord_name:
    :return:
    """
    data[np.isnan(data)] = 0
    labels = np.array([np.argwhere(lab == np.unique(labels_str))[0][0] for lab in labels_str])

    train_inds, test_inds, _ = split_train_test(labels)

    data_train = data[train_inds]
    data_test = data[test_inds]
    labels_train = labels[train_inds]
    labels_test = labels[test_inds]

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    values = {
        "train": {
            'data': data_train,
            'labels': labels_train,
        },
        "tests": {
            'data': data_test,
            'labels': labels_test,
        },
    }

    get_results(values['train'], PCA, ord_name, ax1, 'o')
    get_results(values['tests'], PCA, ord_name, ax1, 'x')

    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/pca_{name}.png", dpi=100)
    plt.close()


def get_results(values, model, name, ax1, marker):
    """
    Gets results

    :param values:
    :param model:
    :param name:
    :param ax1:
    :param marker:
    :return:
    """

    if len(set(values['labels'])) > 2:
        ordin = model(n_components=2)
    else:
        ordin = model(n_components=1)
    principal_components = ordin.fit_transform(values['data'], values['labels'])

    if len(ordin.explained_variance_ratio_) > 1:
        principal_components = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    else:
        principal_components = pd.DataFrame(data=principal_components, columns=['PC1'])

    final_df = pd.DataFrame(
        np.concatenate((principal_components.values, values['labels']), axis=1),
        columns=list(principal_components.columns) + ['label'])

    ax1.set_xlabel(
        f'Component_1 : {np.round(ordin.explained_variance_ratio_[0] * 100, decimals=2)}%',
        fontsize=15
    )
    if len(ordin.explained_variance_ratio_) > 1:
        ax1.set_ylabel(
            f'Component_2 : {np.round(ordin.explained_variance_ratio_[1] * 100, decimals=2)}%',
            fontsize=15
        )

    ax1.set_title(f'2 component {name}', fontsize=20)

    lists = {
        'colors': [],
        'data1': [],
        'data2': []
    }
    for i, target in enumerate(reversed(list(set(values['labels'])))):
        indices_to_keep = [bool(x == target) for x in list(final_df.label)]
        lists['data1'] += [list(final_df.loc[indices_to_keep, 'principal component 1'])]
        if len(ordin.explained_variance_ratio_) > 1:
            lists['data2'] += [list(final_df.loc[indices_to_keep, 'principal component 2'])]
            try:
                assert np.sum(np.isnan(lists['data1'][-1])) == 0 and \
                       np.sum(np.isnan(lists['data2'][-1])) == 0
            except AssertionError:
                print("Nans were detected. Please verify the DataFrame...")
                sys.exit()
            lists['colors'] += [np.array(
                [[
                     plt.get_cmap('coolwarm')(np.linspace(0, 1, len(set(values['labels']))))[i]
                 ] * len(lists['data1'][-1])]
            )]
        else:
            lists['data2'] = False
            lists['colors'] += [np.array([[["g", "b", "k", "r"][i]] * len(lists['data1'][-1])])]

    lists['data1'] = np.hstack(lists['data1']).reshape(-1, 1)
    lists['colors'] = np.hstack(lists['colors']).squeeze()
    if len(lists['data2']) > 0:
        lists['data2'] = np.hstack(lists['data2']).reshape(-1, 1)
        lists['data_colors_vector'] = np.concatenate((lists['data1'],
                                                      lists['data2'],
                                                      lists['colors']), axis=1)

        ax1.scatter(lists['data_colors_vector'][:, 0], lists['data_colors_vector'][:, 1],
                    s=10, alpha=0.5, c=lists['data_colors_vector'][:, 2:],
                    label=values['labels'], marker=marker)
        custom_lines = [Line2D([0], [0], color=plt.get_cmap('coolwarm')(x), lw=4)
                        for x in np.linspace(0, 1, len(set(values['labels'])))]
        ax1.legend(custom_lines, list(set(values['labels'])))

    else:
        ax1.scatter(lists['data1'],
                    rand_jitter(np.zeros_like(lists['data1'])),
                    s=20, alpha=0.5, c=np.array(lists['colors']),
                    label=values['labels'], marker=marker)


def lda(data, labels_str, name, ord_name):
    """

    :param data:
    :param labels_str:
    :param name:
    :param ord_name:
    :return:
    """
    data[np.isnan(data)] = 0
    labels = np.array([np.argwhere(lab == np.unique(labels_str))[0][0] for lab in labels_str])

    train_inds, test_inds, _ = split_train_test(labels)
    # all_train_indices = [s for s, lab in enumerate(categories) if lab in train_inds]
    # test_indices = [s for s, lab in enumerate(categories) if lab in test_inds]

    data_train = data[train_inds]
    data_test = data[test_inds]
    labels_train = labels[train_inds]
    labels_test = labels[test_inds]

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    values = {
        "train": {
            'data': data_train,
            'labels': labels_train,
        },
        "tests": {
            'data': data_test,
            'labels': labels_test,
        },
    }
    get_results(values['train'], LDA, ord_name, ax1, 'o')
    get_results(values['tests'], LDA, ord_name, ax1, 'x')

    os.makedirs("results/images/", exist_ok=True)
    plt.savefig(fname=f"results/images/lda_{name}.png", dpi=100)
    plt.close()


def get_results2(data, labels, ordination, name, ax, marker, binary, test=False):
    cats = np.array([d.split('_')[0] for d in labels])
    reg = re.compile(r'\d')
    labels = np.array([d.split('_')[0] if "blk_p" not in d else reg.split(d)[0] for d in labels])
    if binary:
        if 'blk' in cats:
            for i, cat in enumerate(cats):
                if cat != 'blk':
                    cats[i] = 'not blk'
                else:
                    cats[i] = 'blk'

    if binary:
        if 'blk' in labels:
            for i, label in enumerate(labels):
                if 'blk' not in label:
                    labels[i] = 'not blk'

    unique_cats = []
    for cat in cats:
        if cat not in unique_cats:
            unique_cats += [cat]
    unique_cats = np.sort(np.array(unique_cats))

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels += [label]
    unique_labels = np.sort(np.array(unique_labels))

    if not test:
        if "PCA" in name:
            principal_components = ordination.fit_transform(data)
        else:
            principal_components = ordination.fit_transform(data, cats)

    else:
        principal_components = ordination.transform(data)

    try:
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
    except:
        principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1'])

    labels_df = pd.DataFrame(labels)
    labels_df.index = principal_df.index
    final_df = pd.DataFrame(np.concatenate((principal_df.values, labels_df.values), axis=1),
                            columns=list(principal_df.columns) + ['label'])

    ev = ordination.explained_variance_ratio_
    pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
    ax.set_xlabel(pc1, fontsize=15)
    try:
        pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
        ax.set_ylabel(pc2, fontsize=15)
        ax.set_title(f'2 component {name}', fontsize=20)
    except:
        ax.set_title(f'1 component {name}', fontsize=20)

    cmap = plt.cm.tab20

    num_targets = len(unique_cats)

    cols = cmap(np.linspace(0, 1, num_targets))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)

    colors_list = []
    data1_list = []
    data2_list = []
    new_labels = []
    new_cats = []
    blk_t = 0

    t = 0
    data1_blk = []
    data2_blk = []
    for target in unique_labels:
        indices_to_keep = [True if x == target else False for x in
                           list(final_df.label)]  # 0 is the name of the column with target values
        data1 = list(final_df.loc[indices_to_keep, 'principal component 1'])
        new_labels += [target for _ in range(len(data1))]
        if target == "blk_p":
            target = "blk"
        new_cats += [target for _ in range(len(data1))]

        try:
            data2 = list(final_df.loc[indices_to_keep, 'principal component 2'])
        except:
            data2 = False
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        if target == "blk":
            blk_t += 1
            data1_blk.extend(data1)
            try:
                data2_blk.extend(data2)
            except:
                pass
        else:
            t += 1
        data1_list += [data1]
        if data2:
            data2_list += [data2]
            colors_list += [np.array([[cols[t]] * len(data1)])]
        else:
            colors_list += [np.array([[cols[t]] * len(data1)])]
        if not test and len(data1) > 1:
            if target != "blk":
                try:
                    confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5, edgecolor=cols[t])
                except:
                    pass
            elif blk_t == 2:
                try:
                    confidence_ellipse(np.array(data1_blk), np.array(data2_blk), ax, 1.5, edgecolor=cols[t])
                except:
                    pass

    data1_vector = np.hstack(data1_list).reshape(-1, 1)
    colors_vector = np.hstack(colors_list).squeeze()
    if len(data2_list) > 0:
        data2_vector = np.hstack(data2_list).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]
    else:
        col = np.array(colors_vector)
        data1 = data1_vector

    edgescol = ["k" if cat in ["kox", "sau", "blk", "pae", "sep"] else "w" for cat in new_labels]
    if len(data2_list) > 0:
        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_cats, marker=marker, edgecolors=edgescol)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, num_targets)]
        ax.legend(custom_lines, unique_cats)
    else:
        data1 = data1.reshape(-1)
        zeros = np.zeros_like(data1)
        ax.scatter(data1.reshape(-1), rand_jitter(zeros), s=50, alpha=1.0, c=col, label=new_labels,
                   marker=marker, edgecolors=edgescol)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, num_targets)]
        ax.legend(custom_lines, unique_labels)
    return ordination


# same as get_results2, but ellispe in test mode
def get_results3(data, labels, ordination, name, ax, marker, binary, batches=None, test=False):
    cats = np.array([d.split('_')[0] for d in labels])
    reg = re.compile(r'\d')
    labels = np.array([d.split('_')[0] if "blk_p" not in d else reg.split(d)[0] for d in labels])
    if binary:
        if 'blk' in cats:
            for i, cat in enumerate(cats):
                if cat != 'blk':
                    cats[i] = 'not blk'
                else:
                    cats[i] = 'blk'

    if binary:
        if 'blk' in labels:
            for i, label in enumerate(labels):
                if 'blk' not in label:
                    labels[i] = 'not blk'

    unique_cats = []
    for cat in cats:
        if cat not in unique_cats:
            unique_cats += [cat]
    unique_cats = np.sort(np.array(unique_cats))

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels += [label]
    unique_labels = np.sort(np.array(unique_labels))

    if not test:
        if "PCA" in name:
            principal_components = ordination.fit_transform(data)
        else:
            principal_components = ordination.fit_transform(data, cats)

    else:
        principal_components = ordination.transform(data)

    try:
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
    except:
        principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1'])

    labels_df = pd.DataFrame(labels)
    labels_df.index = principal_df.index
    final_df = pd.DataFrame(np.concatenate((principal_df.values, labels_df.values), axis=1),
                            columns=list(principal_df.columns) + ['label'])
    euclidean = 0
    if batches is not None:
        for b1 in np.unique(batches)[:-1]:
            for b2 in np.unique(batches)[1:]:
                euclidean += cdist(final_df[batches == b1].values[:, :-1], final_df[batches == b2].values[:, :-1]).sum()
    if name != 'FastICA':
        ev = ordination.explained_variance_ratio_
        pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
    else:
        pc1 = 'Component_1'
        pc2 = 'Component_2'
    ax.set_xlabel(pc1, fontsize=15)
    try:
        if name != 'FastICA':
            pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
        ax.set_ylabel(pc2, fontsize=15)
        ax.set_title(f'2 component {name} (batches euclidean dist: {euclidean})', fontsize=20)
    except:
        ax.set_title(f'1 component {name} (batches euclidean dist: {euclidean})', fontsize=20)

    cmap = plt.cm.tab20

    num_targets = len(unique_cats)

    cols = cmap(np.linspace(0, 1, num_targets))
    colors = rcParams['axes.prop_cycle'] = cycler(color=cols)

    colors_list = []
    data1_list = []
    data2_list = []
    new_labels = []
    new_cats = []
    blk_t = 0

    t = 0
    data1_blk = []
    data2_blk = []
    for target in unique_labels:
        indices_to_keep = [True if x == target else False for x in
                           list(final_df.label)]  # 0 is the name of the column with target values
        data1 = list(final_df.loc[indices_to_keep, 'principal component 1'])
        new_labels += [target for _ in range(len(data1))]
        if target == "blk_p":
            target = "blk"
        new_cats += [target for _ in range(len(data1))]

        try:
            data2 = list(final_df.loc[indices_to_keep, 'principal component 2'])
        except:
            data2 = False
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        if target == "blk":
            blk_t += 1
            data1_blk.extend(data1)
            try:
                data2_blk.extend(data2)
            except:
                pass
        else:
            t += 1
        data1_list += [data1]
        if data2:
            data2_list += [data2]
            colors_list += [np.array([[cols[t]] * len(data1)])]
        else:
            colors_list += [np.array([[cols[t]] * len(data1)])]
        if len(data1) > 1:
            if target != "blk":
                try:
                    confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5, edgecolor=cols[t])
                except:
                    pass
            elif blk_t == 2:
                try:
                    confidence_ellipse(np.array(data1_blk), np.array(data2_blk), ax, 1.5, edgecolor=cols[t])
                except:
                    pass

    data1_vector = np.hstack(data1_list).reshape(-1, 1)
    colors_vector = np.hstack(colors_list).squeeze()
    if len(data2_list) > 0:
        data2_vector = np.hstack(data2_list).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]
    else:
        col = np.array(colors_vector)
        data1 = data1_vector

    edgescol = ["k" if cat in ["kox", "sau", "blk", "pae", "sep"] else "w" for cat in new_labels]
    if len(data2_list) > 0:
        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_cats, marker=marker, edgecolors=edgescol)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, num_targets)]
        ax.legend(custom_lines, unique_cats)
    else:
        data1 = data1.reshape(-1)
        zeros = np.zeros_like(data1)
        ax.scatter(data1.reshape(-1), rand_jitter(zeros), s=50, alpha=1.0, c=col, label=new_labels,
                   marker=marker, edgecolors=edgescol)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, num_targets)]
        ax.legend(custom_lines, unique_labels)
    return ordination
