#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""
import sklearn.preprocessing
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from msml.dl.models.pytorch.utils.dataset import MSDataset, load_checkpoint, MSCSV
from msml.dl.models.pytorch.stacked_autoencoder import StackedAutoEncoder
from msml.utils.utils import get_unique_labels
from msml.utils.logging import TensorboardLoggingSDAE
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import LinearSVC
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import itertools
import os
from torch.autograd import Function
from torch import nn
import numpy as np
import pandas as pd
import torch
import json
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def random_init_layer(layer, init_func=nn.init.kaiming_uniform_):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        init_func(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.zero_()

    return layer


def random_init_model(model, init_func=nn.init.kaiming_uniform_):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            init_func(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def hierarchical_labels():
    dict = {
        'ssa': [0, 0, 0],  # Staphylococcus saprophyticus
        'sau': [0, 0, 1],  # Staphylococcus aureus
        'sep': [0, 0, 2],  # Staphylococcus epidermidis
        'sha': [0, 0, 3],  # Staphylococcus haemolyticus

        'smi': [0, 1, 0, 0],  # Streptococcus mitis
        'sag': [0, 1, 0, 1],  # Streptococcus agalactiae

        'efa': [0, 1, 1],  # Enterococcus faecalis

        'pae': [1, 0],  # Pseudomonas aeruginosa

        'pmi': [1, 1, 0],
        'kpn': [1, 1, 1, 0, 0],  # Klebsiella pneumoniae
        'kox': [1, 1, 1, 0, 1],  # Klebsiella oxytoca
        'kae': [1, 1, 1, 0, 2],  # Klebsiella aerogenes
        'eco': [1, 1, 1, 1, 0],  # Escherichia coli
        'ecl': [1, 1, 1, 1, 1],  # Enterobacter cloacae
        'cfr': [1, 1, 1, 1, 2],  # Citrobacter freundii
    }

    return dict


def hierarchical_predictions():
    pass


def log_boxplots(logger, raw, labels):
    norm_data = np.stack([dat / dat.max() for dat in raw.copy()])
    for data, norm in zip([raw, norm_data], ['raw', 'normalized']):
        figure = plt.figure(figsize=(8, 8))
        plt.boxplot(x=data.reshape(-1))
        logger.add_figure(f'boxplots/{norm}/Total boxplot', figure)

        figure.clear()
        figure = plt.figure(figsize=(8, 8))
        flat_data = data.reshape(data.shape[0], -1)
        plt.boxplot(x=flat_data.T)
        logger.add_figure(f'boxplots/{norm}/Samples boxplots', figure)

        figure.clear()
        figure = plt.figure(figsize=(8, 8))
        plt.boxplot(x=data.sum(2))
        logger.add_figure(f'boxplots/{norm}/rt total', figure)

        figure.clear()
        figure = plt.figure(figsize=(8, 8))
        plt.boxplot(x=data.sum(1))
        logger.add_figure(f'boxplots/{norm}/mz total', figure)

        figure.clear()
        figure = plt.figure(figsize=(8, 8))
        plt.boxplot(x=data.sum(2).T)
        logger.add_figure(f'boxplots/{norm}/samples rt', figure)

        figure.clear()
        figure = plt.figure(figsize=(8, 8))
        plt.boxplot(x=data.sum(1).T)
        logger.add_figure(f'boxplots/{norm}/samples mz', figure)


def log_rt_boxplots(data):
    pass


def reverse_output(output, model):
    input = output
    return input


def plot_confusion_matrix(cm, class_names):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(logger, epoch, preds, labels, group='train'):
    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    figure = plot_confusion_matrix(cm, class_names=get_unique_labels(labels))
    logger.add_figure(f"Confusion Matrix {group}", figure, epoch)
    del preds, labels, cm, figure


class Train:
    def __init__(self, path, model, model_name='resnet18', get_data_function=MSCSV, n_channels=1, verbose=1, cv=5,
                 n_epochs=100, batch_size=5, epochs_per_checkpoint=1, aeopt='mse', save=True, load=True, early_stop_val=5000,
                 epochs_per_print=1, test=False, binarize=True, resize=False, train_high_only=False,
                 remove_padding=False, scaler='l2', domain_penalty=True):

        process = get_data_function(path, scaler, binarize, resize, test)
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        data = pool.map(process.process, range(process.__len__()))
        self.data, self.labels, self.lows, self.batches = [x[0] for x in data], pd.Series(
            [x[1] for x in data]), pd.Series(
            [x[2] for x in data]), pd.Series([x[3] for x in data])
        pool.close()
        pool.join()
        pool.terminate()
        del pool, data

        self.cats = self.labels.astype("category").cat.codes.values

        self.nb_classes = len(np.unique(self.cats))
        self.input_shape = [n_channels, self.data[0].shape[0], self.data[0].shape[1]]
        self.model_name = model_name
        self.cv = cv
        self.aeopt = aeopt
        self.verbose = 1
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.remove_padding = remove_padding
        self.optimizer_type = None
        self.scheduler = None
        self.learning_rate = None
        self.weight_decay = None
        self.beta1 = None
        self.domain_penalty = domain_penalty
        self.beta2 = None
        self.min_lr = None
        self.l1 = None
        self.scaler = scaler
        self.binarize = binarize
        self.save = save
        self.load = load
        self.path = path
        self.epochs_per_print = epochs_per_print
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.early_stop = early_stop_val
        self.train_high_only = train_high_only
        self.model = model

    def train(self, params):
        optimizer = None
        self.optimizer_type = 'adam'
        self.learning_rate = 7.618e-05
        self.weight_decay = 3.1e-07
        self.l1 = 0.0046845

        self.dropout = 0.019
        self.beta1 = 0.96662
        self.beta2 = 0.99218
        self.min_lr = 3.6248107962832746e-10
        logreg = self.model(C=params['C'])
        translate_h = 0
        translate_v = 0
        shear = 0
        scale_ratio_max = 0
        scale_erasing_max = 0
        crop_ratio = 0
        crop_scale = 0
        blur_max = 0

        blur_min = 0.01

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            # transforms.Normalize(0.107, 0.253),
        ])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            # transforms.Normalize(0.107, 0.253),
        ])


        epoch = 0
        epoch_offset = max(1, epoch)
        beta = 0.001
        # Get shared output_directory ready
        log_path = f'logs/vaecnn/{self.scaler}/b{beta}/dop{self.domain_penalty}/aeopt{self.aeopt}/' \
                   f'bin{self.binarize}/h{self.train_high_only}/' \
                   f'rempad{self.remove_padding}/' \
                   f'opt{self.optimizer_type}/d{self.dropout}/' \
                   f'trans_{translate_h}_{translate_v}/shear{shear}/blur_{blur_max}_{blur_min}/' \
                   f'cscal{crop_scale}/cratio{crop_ratio}/escal{scale_erasing_max}/' \
                   f'eratio{scale_ratio_max}/lr{self.learning_rate}/wd{self.weight_decay}/' \
                   f'l1{self.l1}'

        os.makedirs(f'{log_path}', exist_ok=True)
        logger_train_rec = SummaryWriter(
            f'{log_path}/rect'
        )
        hparams_filepath = log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        tb_logging = TensorboardLoggingSDAE(hparams_filepath, params)

        data = np.stack(self.data)
        cnn = StackedAutoEncoder(binary=self.binarize, optimizer=params['optimizer'],
                                 lr=self.learning_rate, n_batches=len(set(self.batches)),
                                 nb_classes=self.nb_classes, aeopt=self.aeopt, add_noise=False,
                                 variational=False).to(device)
        nparams = sum(p.numel() for p in cnn.parameters())
        print(f"{nparams} parameters")
        for param in cnn.parameters():
            param.requires_grad = False
        # weight_decay = float(str(self.weight_decay)[:1] + str(self.weight_decay)[-4:])
        # l1 = float(str(self.l1)[:1] + str(self.l1)[-4:])
        learning_rate = float(str(self.learning_rate)[:1] + str(self.learning_rate)[-4:])
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params=cnn.parameters(),
                                         lr=learning_rate,
                                         weight_decay=self.weight_decay,
                                         betas=(self.beta1, self.beta2)
                                         )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params=cnn.parameters(),
                                        lr=learning_rate,
                                        weight_decay=self.weight_decay,
                                        momentum=0.9)
        elif self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(params=cnn.parameters(),
                                            lr=learning_rate,
                                            weight_decay=self.weight_decay,
                                            momentum=0.9)
        else:
            exit('error: no such optimizer type available')

        if self.load:
            cnn, optimizer, epoch, losses, kl_divs, losses_recon, best_loss = load_checkpoint(checkpoint_path='logs/sdaecnn/max/dop0/aeoptmse/bin0/h0/rempad0/optadam/d0.084/trans_0.008_0.176/shear0.065/blur_0.044_0.01/cscal0.571/cratio0.298/escal0.273/eratio2.266/lr5.859e-05/wd3.1e-07/l10.0046845',
                                                                                              model=cnn,
                                                                                              optimizer=optimizer,
                                                                                              name=f"sdaecnn.pth")

        cnn.to(device)
        cnn = cnn.eval()

        # StratifiedKFold not for cross validation, but to split the dataset
        # (break at the end to prevent doing all splits)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        splitter = skf.split(self.data, ['_'.join([str(cat), str(low)]) for cat, low in zip(self.cats, self.lows)])
        all_samples, test_samples = splitter.__next__()
        all_data, test_data = data[all_samples], data[test_samples]
        all_labels, test_labels = self.labels[all_samples].values, self.labels[test_samples].values
        all_cats, test_cats = self.cats[all_samples], self.cats[test_samples]
        all_lows, test_lows = self.lows[all_samples].values, self.lows[test_samples].values
        all_batches, test_batches = self.batches[all_samples].values, self.batches[test_samples].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for i, (train_inds, valid_inds) in enumerate(skf.split(all_data, ['_'.join([str(cat), str(low)]) for cat, low in zip(all_cats, all_lows)])):
            train_inds = train_inds
            train_data, valid_data = all_data[train_inds], all_data[valid_inds]
            train_labels, valid_labels = all_labels[train_inds], all_labels[valid_inds]
            train_cats, valid_cats = all_cats[train_inds], all_cats[valid_inds]
            train_lows, valid_lows = all_lows[train_inds], all_lows[valid_inds]
            train_batches, valid_batches = all_batches[train_inds], all_batches[valid_inds]

            best_loss = -1
            train_set = MSDataset(train_data, train_labels, train_cats, train_lows, train_batches,
                                  transform=transform_train, crop_size=-1,
                                  quantize=False, remove_paddings=self.remove_padding, device='cuda')
            valid_set = MSDataset(valid_data, valid_labels, valid_cats, valid_lows, valid_batches,
                                  transform=transform_valid, crop_size=-1,
                                  quantize=False, device='cuda')
            test_set = MSDataset(test_data, test_labels, test_cats, test_lows, test_batches,
                                 transform=transform_valid, crop_size=-1,
                                 quantize=False, device='cuda')

            train_loader = DataLoader(train_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      pin_memory=False,
                                      drop_last=True)
            valid_loader = DataLoader(valid_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      pin_memory=False,
                                      drop_last=True)
            test_loader = DataLoader(test_set,
                                     num_workers=0,
                                     shuffle=True,
                                     batch_size=self.batch_size,
                                     pin_memory=False,
                                     drop_last=True)

            best_values = {
                'train_acc': [],
                'valid_acc': [],
                'test_acc': [],
            }

            best_epoch = False

            plt.close()

            cats_list = []
            encs_list = []
            labels_list = []
            # scores_list = []
            for i, batch in enumerate(train_loader):
                data, labels, cats, lows, domain = batch
                data[torch.isnan(data)] = 0
                data = data.to(device).float().detach()
                enc, rec, losses, kld = cnn(data)
                encs_list += [enc.detach().cpu().numpy()]
                cats_list.extend(cats)
                labels_list.extend(labels)
                # scores_list += [1 if x.item() == y else 0 for x, y in zip(out.argmax(1), cats)]
                del losses, cats, data, batch, rec, enc
                # fig = plt.figure(figsize=(12, 18))

            # traces['train']['kld'] += [np.mean(values['train']['kld'])]

            encs = np.concatenate(encs_list).squeeze()
            cats_list = torch.stack(cats_list).detach().cpu().numpy()
            logreg.fit(encs, cats_list)
            best_values['train_acc'] += [logreg.score(encs, cats_list)]
            # logger.add_scalar('acc_train', np.mean(scores_list), epoch)
            cats_list = []
            encs_list = []
            labels_list = []
            # scores_list = []
            for i, batch in enumerate(valid_loader):
                data, labels, cats, lows, domain = batch
                data[torch.isnan(data)] = 0
                data = data.to(device).float().detach()
                enc, rec, losses, kld = cnn(data)
                encs_list += [enc.detach().cpu().numpy()]
                cats_list.extend(cats)
                labels_list.extend(labels)
                del losses, cats, data, batch, rec, enc, lows, domain
            encs = np.concatenate(encs_list).squeeze()
            cats_list = torch.stack(cats_list).detach().cpu().numpy()
            best_values['valid_acc'] += [logreg.score(encs, cats_list)]
            # logger.add_scalar('acc_valid', np.mean(scores_list), epoch)
            cats_list = []
            encs_list = []
            labels_list = []
            # scores_list = []
            for i, batch in enumerate(test_loader):
                data, labels, cats, lows, domain = batch
                data[torch.isnan(data)] = 0
                data = data.to(device).float().detach()
                enc, rec, losses, kld = cnn(data)
                encs_list += [enc.detach().cpu().numpy()]
                cats_list.extend(cats)
                labels_list.extend(labels)
                del losses, cats, data, batch, rec, enc, lows, domain
            # traces['test']['losses'] += [np.mean(values['test']['losses'])]
            # traces['test']['kld'] += [np.mean(values['test']['kld'])]
            encs = np.concatenate(encs_list).squeeze()
            cats_list = torch.stack(cats_list).detach().cpu().numpy()
            # linsvc.fit(encs, cats_list)
            # lda.fit(encs, cats_list)
            best_values['test_acc'] += [logreg.score(encs, cats_list)]

            torch.cuda.empty_cache()
        best_values = {key: np.mean(best_vals) for key, best_vals in zip(best_values.keys(), best_values.values())}
        print(best_values)
        # tb_logging.logging(best_values)
        return best_values['valid_acc']

    def print_parameters(self):
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'n_epochs: ' + str(self.n_epochs) + "\n\t",
                  'learning_rate: ' + self.learning_rate.__format__('e') + "\n\t",
                  'weight_decay: ' + self.weight_decay.__format__('e') + "\n\t",
                  'l1: ' + self.l1.__format__('e') + "\n\t",
                  'l2: ' + self.l2.__format__('e') + "\n\t",
                  'optimizer_type: ' + self.optimizer_type + "\n\t",
                  )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="onall/remove_zeros_onall/csv",
                        help="Path")
    parser.add_argument("--test", type=int, default=0,
                        help="Path")
    parser.add_argument("--mz_bin", type=str, default="1")
    parser.add_argument("--domain_penalty", type=int, default=1)
    parser.add_argument("--remove_padding", type=int, default=0)
    parser.add_argument("--rt_bin", type=str, default="1")
    parser.add_argument("--spd", type=str, default="300")
    parser.add_argument("--train_high_only", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="vaecnn_logreg")
    parser.add_argument("--binarize", type=int, default=0)
    parser.add_argument("--scaler", type=str, default='l2')
    parser.add_argument("--input_dir_path", type=str, default='resources/matrices')
    parser.add_argument("--aeopt", type=str, default='mse')
    args = parser.parse_args()
    model = LinearSVC
    train = Train(f"resources/matrices/mz{args.mz_bin}/rt{args.rt_bin}/{args.spd}spd/{args.img_dir}",
                  model, model_name=args.model_name, test=args.test, n_epochs=2000,
                  train_high_only=args.train_high_only, binarize=args.binarize,
                  remove_padding=args.remove_padding, scaler=args.scaler,
                  domain_penalty=args.domain_penalty, aeopt=args.aeopt)

    # train.train()
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "scheduler", "type": "choice", "values": ['ReduceLROnPlateau', 'ReduceLROnPlateau']},
            {"name": "optimizer", "type": "choice", "values": ['adam', 'adam']},
            {"name": "scaler", "type": "choice", "values": ['binarize', 'max', 'maxmax', 'minmax', 'l2']},
            {"name": "beta1", "type": "range", "bounds": [0.9, 0.99], "log_scale": True},
            {"name": "beta2", "type": "range", "bounds": [0.99, 0.9999], "log_scale": True},
            {"name": "min_lr", "type": "range", "bounds": [1e-15, 1e-6], "log_scale": True},
            {"name": "l1", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            # {"name": "momentum", "type": "choice", "values": [0, 0]},

            {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "C", "type": "range", "bounds": [1e-5, 1e3], "log_scale": True},

            {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-4], "log_scale": True},
        ],
        evaluation_function=train.train,
        objective_name='loss',
        minimize=True,
        total_trials=1000,
    )

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))

    # cv_results = cross_validate(model)
    # render(interact_cross_validation(cv_results))
