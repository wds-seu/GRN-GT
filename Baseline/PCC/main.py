import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from dataset import load_dataset, load_node_epr
from sklearn.metrics import roc_auc_score, average_precision_score


# Parse args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser.add_argument('--net', type=str, default='STRING', help='network type')
parser.add_argument('--data', type=str, default='mHSC-L', help='data type')
parser.add_argument('--num', type=int, default=500, help='network scale')
args = parser.parse_args()
print(args)

epr = load_node_epr(os.getcwd() + '/Dataset/' + args.net + ' Dataset/' + args.data + '/TFs+' + str(args.num) + '/')
train_dataset, val_dataset, test_dataset = load_dataset(os.getcwd() + '/train_validation_test/' + args.net + '/' + args.data + ' ' + str(args.num))


def calculate_pcc(data):
    scores = []
    for u, v in zip(data['source'].numpy(), data['target'].numpy()):
        feat_u = epr[u]
        feat_v = epr[v]
        corr = np.corrcoef(feat_u, feat_v)[0, 1]
        scores.append(corr if not np.isnan(corr) else 0.0)
    return np.array(scores)


# train_scores = calculate_pcc(train_dataset.get_all_samples())
# train_labels = train_dataset.get_all_samples()['label']

# val_scores = calculate_pcc(val_dataset.get_all_samples())
# val_labels = val_dataset.get_all_samples()['label']

test_scores = calculate_pcc(test_dataset.get_all_samples())
test_labels = test_dataset.get_all_samples()['label']

# test_scores = F.sigmoid(torch.tensor(test_scores))


auroc = roc_auc_score(test_labels, test_scores)
auprc = average_precision_score(test_labels, test_scores)
print(f"AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}")
