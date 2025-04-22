from graph_transformer import *


def parse_method(args, d, device):
    model = LinkPredictor(d, args.hidden_channels, args.out_channels, args.decoding, args.num_layers, args.num_heads, args.dropout).to(device)
    return model


class BalancedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
        loss = - (self.pos_weight * targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        return loss.mean()


def get_loss_fn(args, train_dataset):
    if args.loss_fn == "bce":
        return nn.BCELoss()
    elif args.loss_fn == "balanced_bce":
        labels = train_dataset.labels
        N_pos = (labels == 1).sum().item()
        N_neg = (labels == 0).sum().item()
        pos_weight = N_neg / (N_pos + 1e-7)
        return BalancedBCELoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_fn}")


def parser_add_main_args(parser):
    # 数据集
    parser.add_argument('--net', type=str, default='STRING', help='network type')
    parser.add_argument('--data', type=str, default='hESC', help='data type')
    parser.add_argument('--num', type=int, default=500, help='network scale')

    # 消融试验
    parser.add_argument('--feature_type', type=str, choices=['fusion', 'cdna', 'exp'], default='fusion')
    parser.add_argument('--use_deepimpute', action='store_true')

    # hyper-parameter for training
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_epoch', type=int, default=10, help='how often to print')
    parser.add_argument('--loss_fn', type=str, choices=['bce', 'balanced_bce'], default='balanced_bce')

    # hyper-parameter for graph_transformer
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--out_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for deep methods')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--decoding', type=str, choices=['concat', 'dot'], default='concat')
