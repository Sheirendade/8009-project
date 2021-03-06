import torch.nn as nn
import torch.optim as optim
import argparse

from torch_geometric.nn import GCNConv, SAGEConv

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from attacks import *
from logger import Logger
from CorrectAndSmooth import CorrectAndSmooth

# loading dataset
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
print(dataset)
data = dataset[0]
print(data)

# split thr data
split_idx = dataset.get_idx_split()

# evaluator
evaluator = Evaluator(name='ogbn-arxiv')

train_idx = split_idx['train']
test_idx = split_idx['test']


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


parser = argparse.ArgumentParser(description='OGBN-Arxiv (SAGE)')
parser.add_argument('--step-size', type=float, default=1e-3)
parser.add_argument('-m', type=int, default=3)

args = parser.parse_args()

model = SAGE(data.num_features, hidden_channels=256, out_channels=dataset.num_classes, num_layers=3, dropout=0.5)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = data.to(device)
data.adj_t = data.adj_t.to_symmetric()

x, y = data.x.to(device), data.y.to(device)
train_idx = split_idx['train'].to(device)
val_idx = split_idx['valid'].to(device)
test_idx = split_idx['test'].to(device)
x_train, y_train = x[train_idx], y[train_idx]

adj_t = data.adj_t.to(device)
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow_(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t
AD = adj_t * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


# FLAG
def train_flag(model, data, train_idx, optimizer, device, args):
    y = data.y.squeeze(1)[train_idx]

    def forward(perturb): return model(data.x + perturb, data.adj_t)[train_idx]

    model_forward = (model, forward)

    loss, _ = flag(model_forward, data.x.shape, y, args, optimizer, device, F.nll_loss)

    return loss.item()


@torch.no_grad()
def test(out=None):
    model.eval()

    out = model(data.x, data.adj_t) if out is None else out
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, out


if __name__ == '__main__':
    runs = 10
    logger = Logger(runs)

    for run in range(runs):
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()

        best_val_acc = 0

        for epoch in range(500):
            loss = train()
            #loss = train_flag(model, data, train_idx, optimizer, device, args)

            train_acc, val_acc, test_acc, out = test()
            result = (train_acc, val_acc, test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                y_soft = out.softmax(dim=-1)

            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

            post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=0.5,
                                    num_smoothing_layers=50, smoothing_alpha=0.8,
                                    autoscale=False, scale=1.)

            print('Correct and smooth...')

            # y_soft = post.correct(y_soft, y_train, train_idx, AD)
            y_soft = post.smooth(y_soft, y_train, train_idx, DAD)
            print('Done!')
            train_acc, val_acc, test_acc, _ = test(y_soft)
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            result = (train_acc, val_acc, test_acc)
            logger.add_result(run, result)

    logger.print_statistics()
