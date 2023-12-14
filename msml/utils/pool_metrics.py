import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm
from statistics import NormalDist

norm.cdf(1.96)
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr, f_oneway


# https://stackoverflow.com/questions/32551610/overlapping-probability-of-two-normal-distribution-with-scipy
def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)

    r = np.roots([a, b, c])[0]

    # integrate
    area = norm.cdf(r, m2, std2) + (1. - norm.cdf(r, m1, std1))
    print(norm.cdf(r, m2, std2), (1. - norm.cdf(r, m1, std1)))
    return area


def batches_overlaps_mean(data, batches):
    means = []
    covs = []
    for b in batches:
        means += [np.mean(data[b], 0)]
        covs += [np.std(data[b], 0)]

    areas = []
    for i in range(len(batches)):
        for j, b in enumerate(batches[i + 1:]):
            # areas += [solve(means[i], means[j+1], covs[i], covs[j+1])]
            normal = multivariate_normal(np.array(means[i]), np.diag(covs[i]))
            for x in data[b]:
                areas += [normal.pdf(x)]

    return np.mean(areas)


def get_pcc(train_b0, train_b1, fun):
    pccs = []
    pvals = []
    for x1 in train_b0:
        for x2 in train_b1:
            pcc, pval = fun(x1, x2)
            pccs += [pcc]
            pvals += [pval]

    return (np.mean(pccs), np.std(pccs)), (np.mean(pvals), np.std(pvals))


def get_qc_pcc(data, fun):
    pccs = []
    pvals = []
    for i, x1 in enumerate(data):
        for x2 in data[i + 1:]:
            pcc, pval = fun(x1, x2)
            pccs += [pcc]
            pvals += [pval]

    return (np.mean(pccs), np.std(pccs)), (np.mean(pvals), np.std(pvals))


def qc_euclidean(data, fun):
    dists = []
    for i, x1 in enumerate(data):
        for x2 in data[i + 1:]:
            dist = fun(x1.reshape(1, -1), x2.reshape(1, -1))
            dists += [dist]

    return np.mean(dists)


def get_euclidean(data, fun, group, metric):
    metric[group]['euclidean'] = np.mean(fun(data[group]))
    metric[f'{group}_pool']['euclidean'] = np.mean(fun(data[f'{group}_pool']))

    return metric


def get_batches_overlap_means(data, batches, metric):
    for group in list(batches.keys()):
        if group not in ['pool', 'all']:
            try:
                metric[group]['overlap'] = batches_overlaps_mean(data[group], batches[group])
            except:
                metric[group]['overlap'] = 'NA'

    return metric


def get_batches_euclidean(data, batches, fun, group, metric):
    metric[group]['b_euclidean'] = batches_euclidean(data[group], batches[group], fun)
    metric[f'{group}_pool']['b_euclidean'] = batches_euclidean(data[f'{group}_pool'], batches[f'{group}_pool'], fun)

    return metric


def batches_euclidean(data, batches, fun):
    dists = []
    for i, x1 in enumerate(batches):
        for x2 in batches[i + 1:]:
            dist = fun(data[x1], data[x2])
            dists += [np.mean(dist)]

    return np.mean(dists)


def euclidean(data, fun):
    dists = []
    for i, x1 in enumerate(data):
        for x2 in data[i + 1:]:
            dist = fun(x1.reshape(1, -1), x2.reshape(1, -1))
            dists += [dist]

    return np.mean(dists)


def get_PCC(data, batches, group, metric):
    metric[group] = {}
    coefs, pvals = np.array([]), np.array([])
    pccs = []
    pvals = []
    qc_pccs = []
    qc_pvals = []
    unique_batches = np.unique(batches[group])
    for i, x1 in enumerate(unique_batches):
        for x2 in unique_batches[i + 1:]:
            train_b0 = data[group][[True if x == x1 else False for x in batches[group]],]
            train_b1 = data[group][[True if x == x2 else False for x in batches[group]],]
            pcc, pval = get_pcc(train_b0, train_b1, pearsonr)
            pccs += [pcc]
            pvals += [pval]

    if np.unique(batches[f'{group}_pool']) > 1:
        for i, x1 in enumerate(np.unique(batches[f'{group}_pool'])):
            for x2 in batches[f'{group}_pool'][i + 1:]:
                pool_b0 = data[f'{group}_pool'][[True if x == x1 else False for x in batches[f'{group}_pool']],]
                pool_b1 = data[f'{group}_pool'][[True if x == x2 else False for x in batches[f'{group}_pool']],]

                pcc, pval = get_pcc(pool_b0, pool_b1, pearsonr)
                qc_pccs += [pcc]
                qc_pvals += [pval]

    (qc_pcc_mean_train_total, qc_pcc_std_train_total), (qc_pval_mean_train_total, qc_pval_std_train_total) = get_qc_pcc(
        data[f'{group}_pool'], pearsonr)
    (pcc_mean_train_total, pcc_std_train_total), (pval_mean_train_total, pval_std_train_total) = get_qc_pcc(data[group],
                                                                                                            pearsonr)

    pcc_mean_train = np.mean(pccs)
    pcc_std_train = np.std(pccs)
    pval_mean_train = np.mean(pvals)
    pval_std_train = np.std(pvals)

    qc_pcc_mean_train = np.mean(qc_pccs)
    qc_pcc_std_train = np.std(qc_pccs)
    qc_pval_mean_train = np.mean(qc_pvals)
    qc_pval_std_train = np.std(qc_pvals)

    metric[group]['aPCC_score'] = pcc_mean_train
    metric[group]['aPCC_pval'] = pval_mean_train
    metric[group]['aPCC_score_total'] = pcc_mean_train_total
    metric[group]['aPCC_pval_total'] = pval_mean_train_total
    metric[group]['qc_aPCC_score'] = qc_pcc_mean_train
    metric[group]['qc_aPCC_pval'] = qc_pval_mean_train
    metric[group]['qc_aPCC_score_total'] = qc_pcc_mean_train_total
    metric[group]['qc_aPCC_pval_total'] = qc_pval_mean_train_total

    return metric


def get_qc_euclidean(pool_data, group, metric):
    # qc_dist = qc_euclidean(pool_data, pdist)
    metric[group]['qc_dist'] = np.mean(pdist(pool_data))

    return metric


def log_pool_metrics(name, data, batches, logger, epoch, metrics):
    metric = {}
    for group in ['all', 'train', 'valid', 'test']:
        if f"{group}_pool" not in list(batches.keys()):
            continue
        metric[group] = {}
        metric[f'{group}_pool'] = {}
        batch_train_samples = [[i for i, batch in enumerate(batches[group].tolist()) if batch == b] for b in
                               np.unique(batches[group])]
        batch_pool_samples = [[i for i, batch in enumerate(batches[f"{group}_pool"].tolist()) if batch == b] for b in
                              np.unique(batches[f"{group}_pool"])]

        batches_sample_indices = {
            group: batch_train_samples,
            f'{group}_pool': batch_pool_samples,
        }
        # Average Pearson's Correlation Coefficients
        try:
            metric = get_PCC(data, batches, group, metric)
        except:
            pass

        # QC euclidean distance
        try:
            metric = get_qc_euclidean(data[f'{group}_pool'], group, metric)
        except:
           pass

        # Batch avg distance
        try:
            metric = get_batches_euclidean(data, batches_sample_indices, cdist, group, metric)
        except:
            pass

        # avg distance
        try:
            metric = get_euclidean(data, pdist, group, metric)
        except:
            pass

        # overlap
        # metric = get_batches_overlap_means(name, data, batches_sample_indices, metric)

        # Multivariate ANOVA
        # fit = MANOVA(train_data, [int(b) for b in train_batches])
        # print(fit.mv_test())

        # Multivariate ANOVA (principal components)
        # pca = PCA(n_components=2)
        # pcs_train = pca.fit_transform(train_data)
        # fit = MANOVA(pcs_train, [int(b) for b in train_batches])
        # print(fit.mv_test())
    for group in metric:
        for m in metric[group]:
            if not np.isnan(metric[group][m]):
                logger.add_scalar(f'pool_metrics/{m}/{group}/{name}', metric[group][m], epoch)
    metrics['pool_metrics'][name] = metric

    return metrics
