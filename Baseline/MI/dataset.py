import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GraphDataset:
    def __init__(self, df):
        self.x_edges = torch.tensor(df['TF'].values, dtype=torch.long)
        self.y_edges = torch.tensor(df['Target'].values, dtype=torch.long)
        self.labels = torch.tensor(df['Label'].values, dtype=torch.float32)

    def get_all_samples(self):
        return {
            'source': self.x_edges,
            'target': self.y_edges,
            'label': self.labels
        }


def load_dataset(train_val_test_set_path):
    df = pd.read_csv(train_val_test_set_path + '/Train_set.csv')
    train_dataset = GraphDataset(df)
    df = pd.read_csv(train_val_test_set_path + '/Validation_set.csv')
    val_dataset = GraphDataset(df)
    df = pd.read_csv(train_val_test_set_path + '/Test_set.csv')
    test_dataset = GraphDataset(df)
    return train_dataset, val_dataset, test_dataset


def load_node_epr(epr_csv_path):
    data = pd.read_csv(epr_csv_path + 'BL--ExpressionData.csv', index_col=0)
    epr = data.iloc[:, :].values
    standard = StandardScaler()
    epr = standard.fit_transform(epr)
    return torch.tensor(epr, dtype=torch.float32)
