import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(GraphTransformerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        self.out_channels = out_channels
        self.num_heads = num_heads

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, z):
        B, N = z.size(0), z.size(1)
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels).transpose(1, 2)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels).transpose(1, 2)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels).transpose(1, 2)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.out_channels ** 0.5)  # [B, H, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        z_next = torch.matmul(attn_weights, value)  # [B, H, N, D]
        z_next = z_next.transpose(1, 2).reshape(B, N, -1)  # [B, N, H * D]
        z_next = self.Wo(z_next)
        return z_next


class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, dropout):
        super(GraphTransformer, self).__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                GraphTransformerConv(hidden_channels, hidden_channels, num_heads))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.activation = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        x = x.unsqueeze(0)  # [B, N, H, D]
        z = self.fcs[0](x)
        z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_ = [z]

        for i, conv in enumerate(self.convs):
            z = conv(z)
            z += layer_[i]
            z = self.bns[i + 1](z)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)
        x_out = self.fcs[-1](z).squeeze(0)
        return x_out


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, decoding, num_layers, num_heads, dropout):
        super().__init__()
        self.decoding = decoding
        self.encoder = GraphTransformer(in_channels, hidden_channels, num_layers, num_heads, dropout)
        if decoding == 'concat':
            self.link_predictor = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.link_predictor = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, 1),
                nn.Sigmoid()
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.link_predictor:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, y, node_features):
        node_feat = self.encoder(node_features)
        h_x = node_feat[x]
        h_y = node_feat[y]
        if self.decoding == 'concat':
            h = torch.cat([h_x, h_y], dim=1)
        else:
            h = h_x * h_y
        pred = self.link_predictor(h).squeeze()
        return pred
