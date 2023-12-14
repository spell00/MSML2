import pandas as pd
import json


def find_best_model(infos_json):
    best_acc = -1
    best_model = None
    for model in infos_json:
        if float(infos_json[model]['valid_acc_mean_highs']) > best_acc:
            best_acc = float(infos_json[model]['valid_acc_mean_highs'])
            best_model = model
    return best_model


if __name__ == "__main__":
    mzbins = ['mz0.2']
    rtbins = ['rt20']
    minmzs = ['minmz0.0']
    minrts = ['minrt0.0']
    spds = ['200spd']
    scalers = ['robust']
    scaler_preprocesses = ['none']
    combats = ['combat0_0', 'combat1_0']
    strides = ['stride0']
    thresholds = ['thres0.99']
    run_names = ['plate_1', 'plate_2', 'plate_3', 'plate_4']
    inferences = ['inference0']

    results = pd.DataFrame()

    ind = 0
    for mzbin in mzbins:
        for rtbin in rtbins:
            for minmz in minmzs:
                for minrt in minrts:
                    for spd in spds:
                        for scaler in scalers:
                            for scaler_preprocess in scaler_preprocesses:
                                for combat in combats:
                                    for stride in strides:
                                        for thres in thresholds:
                                            for run_name in run_names:
                                                for inference in inferences:
                                                    ind += 1
                                                    if len(mzbins) > 1:
                                                        results.loc[ind, 'mzbin'] = mzbin
                                                    if len(rtbins) > 1:
                                                        results.loc[ind, 'rtbin'] = rtbin
                                                    if len(minmzs) > 1:
                                                        results.loc[ind, 'minmz'] = minmz
                                                    if len(minrts) > 1:
                                                        results.loc[ind, 'minrt'] = minrt
                                                    if len(spds) > 1:
                                                        results.loc[ind, 'spd'] = spd
                                                    if len(scalers) > 1:
                                                        results.loc[ind, 'scaler'] = scaler
                                                    if len(scaler_preprocesses) > 1:
                                                        results.loc[ind, 'scaler_preprocess'] = scaler_preprocess
                                                    if len(combats) > 1:
                                                        results.loc[ind, 'combat_features'] = combat.split('combat')[1].split('_')[0]
                                                        results.loc[ind, 'combat_vals'] = combat.split('_')[1]
                                                    if len(strides) > 1:
                                                        results.loc[ind, 'stride'] = stride
                                                    if len(thresholds) > 1:
                                                        results.loc[ind, 'thres'] = thres
                                                    if len(run_names) > 1:
                                                        results.loc[ind, 'run_name'] = run_name
                                                    if len(inferences) > 1:
                                                        results.loc[ind, 'inference'] = inference
                                                    path = f'results/{mzbin}/{rtbin}/{minmz}/{minrt}/{spd}/{scaler}/{combat}/{stride}/{scaler_preprocess}/loginloop/corrected0/drop_lowsno/drop_blks0/binary0/boot0/robust/cv5/nrep1/ovr0/{thres}/{run_name}/{inference}/saved_models/sklearn/best_params.json'
                                                    with open(path, 'r') as handle:
                                                        infos_json = json.load(handle)
                                                    best_model = find_best_model(infos_json)

                                                    results.loc[ind, 'best_model'] = best_model
                                                    results.loc[ind, 'valid_acc_mean'] = infos_json[best_model]['valid_acc_mean']
                                                    results.loc[ind, 'valid_acc_mean_highs'] = infos_json[best_model]['valid_acc_mean_highs']
                                                    results.loc[ind, 'valid_acc_mean_lows'] = infos_json[best_model]['valid_acc_mean_lows']
                                                    results.loc[ind, 'test_acc'] = infos_json[best_model]['test_acc']
                                                    results.loc[ind, 'test_acc_highs'] = infos_json[best_model]['test_highs_acc']
                                                    results.loc[ind, 'test_acc_lows'] = infos_json[best_model]['test_lows_acc']

    results.to_csv('summary_table.csv', index=None)
