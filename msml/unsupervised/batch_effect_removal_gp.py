import warnings
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
from msml.scikit_learn.utils import get_scaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef as MCC
from combat.pycombat import pycombat
from skopt import gp_minimize
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from msml.scikit_learn.train.params_gp import *

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src/models/sklearn/'



class Train:
    def __init__(self, model, data, hparams_names, args, ovr, binary=True):
        try:
            with open(f'{args.destination}/saved_models/sklearn/best_params.json', "r") as json_file:
                self.previous_models = json.load(json_file)
        except:
            self.previous_models = {}
        self.random = args.random
        self.model = model
        self.data = data
        self.hparams_names = hparams_names
        # self.train_indices, self.test_indices, _ = split_train_test(self.labels)
        self.n_splits = args.n_splits

        self.n_repeats = args.n_repeats
        self.jackknife = args.jackknife
        self.best_scores_train = None
        self.best_scores_valid = None
        self.best_mccs_train = None
        self.best_mccs_valid = None
        self.scores_train = None
        self.scores_valid = None
        self.mccs_train = None
        self.mccs_valid = None
        self.y_preds = np.array([])
        self.y_valids = np.array([])
        self.top3_valid = None
        self.top3_train = None
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

        # Find why test has 1 more column
        if self.data['train']['data'].shape[1] > self.data['test']['data'].shape[1]:
            self.data['train']['data'] = self.data['train']['data'].iloc[:, :self.data['test']['data'].shape[1]]
        elif self.data['test']['data'].shape[1] > self.data['train']['data'].shape[1]:
            self.data['test']['data'] = self.data['test']['data'].iloc[:, :self.data['train']['data'].shape[1]]
        # Remove features with a ratio of data > 0 that is < a threshold
        # good_features = np.array(
        #     [i for i in range(self.data['train']['data'].shape[1]) if sum(self.data['train']['data'][:, i] != 0) > int(self.data['train']['data'].shape[0] * 0.1)])
        # self.data['train']['data'] = self.data['train']['data'][:, good_features]
        # self.data['test']['data'] = self.data['test']['data'][:, good_features]

        train_blks = self.data['train']['blks']

        train_labels = self.data['train']['labels']
        train_data = self.data['train']['data']
        train_batches = self.data['train']['batches']
        train_highs = self.data['train']['highs']
        train_lows = self.data['train']['lows']

        valid_labels = self.data['valid']['labels']
        valid_data = self.data['valid']['data']
        valid_batches = self.data['valid']['batches']
        valid_highs = self.data['valid']['highs']
        valid_lows = self.data['valid']['lows']

        test_blks = self.data['test']['blks']
        test_labels = self.data['test']['labels']
        test_data = self.data['test']['data']
        test_batches = self.data['test']['batches']

        low_labels = self.data['lows']['labels']
        # low_data = self.data['test']['data']
        # low_batches = self.data['test']['batches']

        unique_labels = []
        for l in train_labels:
            if l not in unique_labels:
                unique_labels += [l]

        unique_labels = np.array(unique_labels)
        train_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in train_labels])
        valid_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in valid_labels])
        test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in test_labels])
        if self.data["lows"]["data"] is not None:
            low_test_classes = np.array([np.argwhere(l == unique_labels)[0][0] for l in low_labels])

        if args.scaler != 'binarize':
            scaler = get_scaler(args.scaler)()
        else:
            from sklearn.preprocessing import Binarizer
            scaler = Pipeline([('minmax', get_scaler('minmax')()), ('binarizer', Binarizer(threshold=threshold))])
        if test_data.shape[1] < 10000:
            cutoff = test_data.shape[1]
        else:
            cutoff = 10000
        if args.correct_batches:
            df_train = train_data.iloc[:, :cutoff]
            df_valid = valid_data.iloc[:, :cutoff]
            df_test = test_data.iloc[:, :cutoff]
            df = pd.concat((df_train, df_valid, df_test), 0).T
            all_batches = np.concatenate((train_batches, valid_batches, test_batches))
            df[df.isna()] = 0
            all_data = pycombat(df, all_batches).T.values

            train_data = pd.DataFrame(all_data[:train_data.shape[0]])
            valid_data = pd.DataFrame(all_data[train_data.shape[0]:train_data.shape[0]+valid_data.shape[0]])
            test_data = pd.DataFrame(all_data[:test_data.shape[0]])

        print(f'Iteration: {self.iter}')

        self.scores_train = []
        self.scores_valid = []
        self.scores_valid_highs = []
        self.scores_valid_lows = []
        self.mccs_train = []
        self.mccs_valid = []
        self.top3_train = []
        self.top3_valid = []
        broke = False
        y_pred_valids = []
        y_pred_valids_lows = []
        y_pred_valids_highs = []
        y_valids = []
        y_valids_highs = []
        y_valids_lows = []
        x_valids = []
        x_valids_highs = []
        x_valids_lows = []
        # train_nums = np.arange(0, len(all_x_train))

        transductive = 0
        # if transductive:
        #     all_x_train = scaler.fit_transform(all_x_train)
        # else:
        #     all_x_train = all_x_train.to_numpy()

        # for i, (train_inds, valid_inds) in enumerate(skf.split(train_nums, all_y_train)):
            # Just plot the first iteration, it will already be crowded if doing > 100 optimization iterations
        # new_nums = [train_indices[i] for i in inds]

        # x_train = all_x_train[train_inds]
        # feats_inds = np.arange(train_data.shape[1])
        # feats_inds = feats_inds[:features_cutoff]
        # x_train = x_train[:, feats_inds]
        # y_train = all_y_train[train_inds]
        # x_valid = all_x_train[valid_inds]
        # x_valid = x_valid[:, feats_inds]
        # y_valid = all_y_train[valid_inds]
        # valid_highs = np.array([int(i) for i, x in enumerate(train_highs[valid_inds]) if x == 1])
        # valid_lows = np.array([int(i) for i, x in enumerate(train_lows[valid_inds]) if x == 1])
        # valid_blks = np.array([int(i) for i, x in enumerate(train_blks[valid_inds]) if x == 1])
        valid_highs = np.concatenate((valid_highs, valid_blks))
        valid_lows = np.concatenate((valid_lows, valid_blks))
        # if not transductive:
        x_train = scaler.fit_transform(train_data.iloc[:, :features_cutoff])
        x_valid = scaler.transform(valid_data.iloc[:, :features_cutoff])

        m = self.model()
        m.set_params(**param_grid)
        if self.ovr:
            m = OneVsRestClassifier(m)
        try:
            m.fit(x_train, train_classes)
        except:
            return 1

        score_valid = m.score(x_valid, valid_classes)
        score_train = m.score(x_train, train_classes)
        print('valid_score:', score_valid, "features_cutoff", features_cutoff, 'h_params:', param_grid)
        self.scores_train += [score_train]
        self.scores_valid += [score_valid]
        try:
            self.scores_valid_highs += [m.score(x_valid[valid_highs], valid_classes[valid_highs])]
            self.scores_valid_lows += [m.score(x_valid[valid_lows], valid_classes[valid_lows])]
        except:
            valid_highs = [int(x) for x in valid_highs]
            valid_lows = [int(x) for x in valid_lows]
            self.scores_valid_highs += [m.score(x_valid[valid_highs], valid_classes[valid_highs])]
            try:
                self.scores_valid_lows += [m.score(x_valid[valid_lows], valid_classes[valid_lows])]
            except:
                pass

        y_pred_train = m.predict(x_train)
        y_pred_valid = m.predict(x_valid)

        y_pred_valids.extend(y_pred_valid)
        y_pred_valids_highs.extend(y_pred_valid[valid_highs])
        y_pred_valids_lows.extend(y_pred_valid[valid_lows])

        y_valids.extend(valid_classes)
        y_valids_highs.extend(valid_classes[valid_highs])
        y_valids_lows.extend(valid_classes[valid_lows])
        x_valids.extend(x_valid)
        x_valids_highs.extend(x_valid[valid_highs])
        x_valids_lows.extend(x_valid[valid_lows])

        mcc_train = MCC(train_classes, y_pred_train)
        mcc_valid = MCC(valid_classes, y_pred_valid)
        self.mccs_train += [mcc_train]
        self.mccs_valid += [mcc_valid]

        try:
            y_proba_train = m.predict_proba(x_train)
            y_proba_valid = m.predict_proba(x_valid)
            y_top3_train = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_train.argsort(1), train_classes)])
            y_top3_valid = np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                    zip(y_proba_valid.argsort(1), valid_classes)])

            self.top3_train += [y_top3_train]
            self.top3_valid += [y_top3_valid]
        except:
            self.top3_train += [-1]
            self.top3_valid += [-1]

        if self.best_scores_valid is None:
            self.best_scores_valid = 0

        score = np.mean(self.scores_valid)
        if np.isnan(score):
            score = 0
        if model not in self.previous_models:
            self.previous_models[model] = {'valid_acc_mean': -1}
        if score > np.mean(self.best_scores_valid) and score > float(self.previous_models[model]['valid_acc_mean']):
            self.best_scores_train = self.scores_train
            self.best_scores_valid = self.scores_valid
            self.best_scores_valid_highs = self.scores_valid_highs
            self.best_scores_valid_lows = self.scores_valid_lows
            self.best_mccs_train = self.mccs_train
            self.best_mccs_valid = self.mccs_valid
            self.best_top3_train = self.top3_train
            self.best_top3_valid = self.top3_valid
            fig = get_confusion_matrix(y_valids, y_pred_valids, unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid", acc=score)
            fig = get_confusion_matrix(y_valids_highs, y_pred_valids_highs, unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid_highs", acc=np.mean(self.best_scores_valid_highs))
            fig = get_confusion_matrix(y_valids_lows, y_pred_valids_lows, unique_labels)
            save_confusion_matrix(fig, f"{args.destination}/confusion_matrices/{model}_valid_lows", acc=np.mean(self.best_scores_valid_lows))
            try:
                self.best_roc_score = save_roc_curve(m, x_valids, y_valids, unique_labels,
                                                     f"{args.destination}/ROC/{model}_valid", binary=args.binary,
                                                     acc=score)
            except:
                pass
            try:
                self.best_roc_score_highs = save_roc_curve(m, x_valids_highs, y_valids_highs, unique_labels,
                                                     f"{args.destination}/ROC/{model}_valid_highs", binary=args.binary,
                                                     acc=np.mean(self.best_scores_valid_highs))
            except:
                pass
            try:
                self.best_roc_score_lows = save_roc_curve(m, x_valids_lows, y_valids_lows, unique_labels,
                                                     f"{args.destination}/ROC/{model}_valid_lows", binary=args.binary,
                                                     acc=np.mean(self.best_scores_valid_lows))
            except:
                pass
        return 1 - score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--transductive', type=float, default=0, help='not functional')
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--n_calls', type=int, default=20)
    parser.add_argument('--log2', type=str, default='inloop', help='inloop or after')
    parser.add_argument('--ovr', type=int, default=1, help='OneVsAll strategy')
    parser.add_argument('--drop_lows', type=str, default="none")
    parser.add_argument('--drop_blks', type=int, default=0)
    parser.add_argument('--correct_batches', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--feature_selection', type=str, default='mutual_info_classif',
                        help='mutual_info_classif or f_classif')
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--spd', type=int, default=200)
    parser.add_argument('--mz', type=float, default=0.2)
    parser.add_argument('--rt', type=float, default=20)
    parser.add_argument('--min_rt', type=float, default=0)
    parser.add_argument('--min_mz', type=float, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--run_name', type=str, default='all')
    parser.add_argument('--scaler', type=str, default='robust')
    parser.add_argument('--preprocess_scaler', type=str, default='none')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--jackknife', type=int, default=0)
    parser.add_argument('--inference_inputs', type=int, default=0)
    parser.add_argument("--input_dir", type=str, help="Path to intensities csv file")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to intensities csv file")
    parser.add_argument("--combat", type=int, default=0)
    parser.add_argument("--combat_data", type=int, default=0)
    parser.add_argument("--stride", type=str, default=0)
    args = parser.parse_args()
    if args.combat == 0:
        args.combat_data = 0
    if int(args.rt) == args.rt:
        args.rt = int(args.rt)
    if int(args.mz) == args.mz:
        args.mz = int(args.mz)

    args.destination = f"{args.output_dir}/mz{args.mz}/rt{args.rt}/minmz{args.min_mz}/minrt{args.min_rt}/" \
                       f"{args.spd}spd/{args.scaler}/combat{args.combat}_{args.combat_data}/stride{args.stride}/" \
                       f"{args.preprocess_scaler}/log{args.log2}/corrected{args.correct_batches}/" \
                       f"drop_lows{args.drop_lows}/drop_blks{args.drop_blks}/binary{args.binary}/" \
                       f"boot{args.jackknife}/{args.scaler}/cv{args.n_splits}/nrep{args.n_repeats}/" \
                       f"ovr{args.ovr}/thres{args.threshold}/{args.run_name}/inference{args.inference_inputs}"

    if args.combat_data and args.combat:
        train_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                           f"stride{args.stride}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/train/{args.run_name}/" \
                           f"BACT_train_inputs_gt0.0_{args.run_name}_combat.csv"
    else:
        train_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                           f"stride{args.stride}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/train/{args.run_name}/" \
                           f"BACT_train_inputs_{args.run_name}.csv"
    valid_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                       f"stride{args.stride}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/valid/{args.run_name}/" \
                       f"BACT_valid_inputs_{args.run_name}.csv"
    test_input = f"{args.input_dir}/mz{args.mz}/rt{args.rt}/{args.spd}spd/combat{args.combat}/" \
                       f"stride{args.stride}/{args.preprocess_scaler}/log{args.log2}/{args.feature_selection}/test/{args.run_name}/" \
                       f"BACT_inference_inputs_{args.run_name}.csv"

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
