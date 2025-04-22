import torch
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
from deepimpute.multinet import MultiNet
from sklearn.preprocessing import StandardScaler


class GraphDataset(Dataset):
    def __init__(self, df):
        self.x_edges = torch.tensor(df['TF'].values, dtype=torch.long)
        self.y_edges = torch.tensor(df['Target'].values, dtype=torch.long)
        self.labels = torch.tensor(df['Label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.x_edges)

    def __getitem__(self, idx):
        return {
            'source': self.x_edges[idx],
            'target': self.y_edges[idx],
            'label': self.labels[idx]
        }


def load_dataset(train_val_test_set_path):
    df = pd.read_csv(train_val_test_set_path + '/Train_set.csv')
    train_dataset = GraphDataset(df)
    df = pd.read_csv(train_val_test_set_path + '/Validation_set.csv')
    val_dataset = GraphDataset(df)
    df = pd.read_csv(train_val_test_set_path + '/Test_set.csv')
    test_dataset = GraphDataset(df)
    return train_dataset, val_dataset, test_dataset


def load_node_epr(use_deepimpute, epr_csv_path):
    data = pd.read_csv(epr_csv_path + 'BL--ExpressionData.csv', index_col=0)
    if use_deepimpute:
        deepimpute_csv = epr_csv_path + 'deepimputed_exp.csv'
        if os.path.exists(deepimpute_csv):
            data = pd.read_csv(deepimpute_csv, index_col=0)
        else:
            deepimpute_model = MultiNet(output_prefix='./deepimpute_model')
            deepimpute_model.fit(data)
            data = deepimpute_model.predict(data)
            data.to_csv(deepimpute_csv)
    epr = data.iloc[:, :].values
    standard = StandardScaler()
    epr = standard.fit_transform(epr)
    return torch.tensor(epr, dtype=torch.float32)
