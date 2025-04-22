import os
import torch
import random
import warnings
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from fetch_gene_sequence import load_gene_cache
from gene_embedding import load_gene_model, gene_embedding
from dataset import GraphDataset, load_dataset, load_node_epr
from parse import parse_method, parser_add_main_args, get_loss_fn
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model_, dataloader_, device_, features_):
    model_.eval()
    preds_ = []
    labels_ = []
    with torch.no_grad():  # 在验证阶段不计算梯度
        for sample_ in dataloader_:
            x_ = sample_['source'].to(device_)
            y_ = sample_['target'].to(device_)
            label_ = sample_['label'].to(device_)
            pred_ = model_(x_, y_, features_)
            preds_.append(pred_.flatten().cpu().numpy())
            labels_.append(label_.flatten().cpu().numpy())
    # 将预测值和标签拼接成一个大数组
    preds_ = np.concatenate(preds_)
    labels_ = np.concatenate(labels_)
    # 计算AUROC和AUPRC
    auroc_ = roc_auc_score(labels_, preds_)
    auprc_ = average_precision_score(labels_, preds_)
    return auroc_, auprc_


# Parse args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
parser.add_argument('--rob', type=float, default=0.1)
args = parser.parse_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

gene_cache = load_gene_cache()
tokenizer, gene_emb_model, gene_emb_dim = load_gene_model(device)
epr_csv_path = os.getcwd() + '/Dataset/' + args.net + ' Dataset/' + args.data + '/TFs+' + str(args.num) + '/'
gene_name = pd.read_csv(epr_csv_path + "BL--ExpressionData.csv").iloc[:, 0:1].values
num_nodes = gene_name.shape[0]
d = 0
epr = None
if args.feature_type == 'fusion' or args.feature_type == 'exp':
    epr = load_node_epr(args.use_deepimpute, epr_csv_path)
    d += epr.shape[1]
if args.feature_type == 'fusion' or args.feature_type == 'cdna':
    gene_embs = []
    gene_emb_model.eval()
    with torch.no_grad():
        for name in gene_name:
            gene_emb = gene_embedding(name[0], tokenizer, gene_emb_model, gene_cache, device)
            gene_embs.append(gene_emb)
        gene_embs = torch.cat(gene_embs, dim=0).to(device)
    d += gene_embs.shape[1]

path = os.getcwd() + '/train_validation_test/' + args.net + '/' + args.data + ' ' + str(args.num)
_, val_dataset, test_dataset = load_dataset(path)
df = pd.read_csv(path + '/Train_set.csv')
subset_df = df.sample(frac=args.rob, random_state=42)
subset_df = subset_df.reset_index(drop=True)
train_dataset = GraphDataset(subset_df)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=16384, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=16384, drop_last=False)

# Load method
model = parse_method(args, d, device)
optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.lr}], weight_decay=args.weight_decay)
loss_fn = get_loss_fn(args, train_dataset)

mean_auroc = 0
mean_auprc = 0
for run in range(args.runs):
    fix_seed(run)
    # Training loop
    model.reset_parameters()
    best_val_auroc = 0
    best_val_auprc = 0
    test_auroc = 0
    test_auprc = 0
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            if batch['source'].shape[0] == 1:
                continue

            optimizer.zero_grad()
            x_batch = batch['source'].to(device)
            y_batch = batch['target'].to(device)
            label_batch = batch['label'].to(device)
            if args.feature_type == 'fusion':
                pred = model(x_batch, y_batch, torch.cat((gene_embs, epr.to(device)), dim=1))
            elif args.feature_type == 'exp':
                pred = model(x_batch, y_batch, epr.to(device))
            else:
                pred = model(x_batch, y_batch, gene_embs)
            loss = loss_fn(pred, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % args.eval_epoch == 0:
            if args.feature_type == 'fusion':
                auroc, auprc = evaluate(model, val_dataloader, device, torch.cat((gene_embs, epr.to(device)), dim=1))
            elif args.feature_type == 'exp':
                auroc, auprc = evaluate(model, val_dataloader, device, epr.to(device))
            else:
                auroc, auprc = evaluate(model, val_dataloader, device, gene_embs)
            print(f"Validation at epoch {epoch + 1}: AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}")
            if best_val_auroc < auroc:
                best_val_auroc = auroc
                if args.feature_type == 'fusion':
                    test_auroc, _ = evaluate(model, test_dataloader, device,
                                             torch.cat((gene_embs, epr.to(device)), dim=1))
                elif args.feature_type == 'exp':
                    test_auroc, _ = evaluate(model, test_dataloader, device, epr.to(device))
                else:
                    test_auroc, _ = evaluate(model, test_dataloader, device, gene_embs)
            if best_val_auprc < auprc:
                best_val_auprc = auprc
                if args.feature_type == 'fusion':
                    _, test_auprc = evaluate(model, test_dataloader, device,
                                             torch.cat((gene_embs, epr.to(device)), dim=1))
                elif args.feature_type == 'exp':
                    _, test_auprc = evaluate(model, test_dataloader, device, epr.to(device))
                else:
                    _, test_auprc = evaluate(model, test_dataloader, device, gene_embs)

    print(f"Test at run {run + 1}: AUROC = {test_auroc:.4f}, AUPRC = {test_auprc:.4f}")
    mean_auroc += test_auroc
    mean_auprc += test_auprc

print(f"mean test result: AUROC = {mean_auroc / args.runs:.4f}, AUPRC = {mean_auprc / args.runs:.4f}")
