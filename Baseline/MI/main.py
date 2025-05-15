import os
import time
import torch
import argparse
import numpy as np
from dataset import load_dataset, load_node_epr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, average_precision_score, mutual_info_score


# Parse args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser.add_argument('--net', type=str, default='STRING', help='network type')
parser.add_argument('--data', type=str, default='mHSC-L', help='data type')
parser.add_argument('--num', type=int, default=500, help='network scale')
args = parser.parse_args()
print(args)

epr = load_node_epr(os.getcwd() + '/Dataset/' + args.net + ' Dataset/' + args.data + '/TFs+' + str(args.num) + '/')
train_dataset, val_dataset, test_dataset = load_dataset(os.getcwd() + '/train_validation_test/' + args.net + '/' + args.data + ' ' + str(args.num))


def calculate_mi(data, k=4):
    mi_scores = []
    discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
    for u, v in zip(data['source'].numpy().flatten(), data['target'].numpy().flatten()):
        x = epr[u]
        y = epr[v]
        x_flat = x.flatten()
        y_flat = y.flatten()
        x_discrete = discretizer.fit_transform(x_flat.reshape(-1, 1)).flatten()
        y_discrete = discretizer.fit_transform(y_flat.reshape(-1, 1)).flatten()
        mi = mutual_info_score(x_discrete, y_discrete)
        mi_scores.append(max(mi, 0.0))
    return np.array(mi_scores)


# train_scores = calculate_mi(train_dataset.get_all_samples())
# train_labels = train_dataset.get_all_samples()['label']

# val_scores = calculate_mi(val_dataset.get_all_samples())
# val_labels = val_dataset.get_all_samples()['label']

test_scores = calculate_mi(test_dataset.get_all_samples())
test_labels = test_dataset.get_all_samples()['label']

auroc = roc_auc_score(test_labels, test_scores)
auprc = average_precision_score(test_labels, test_scores)
print(f"AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}")
