import argparse


import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import numpy as np
import torch.nn as nn
from attacks import *


class SAGE_res(torch.nn.Module):
    def __init__(self, dataset, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_res, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers

        self.input_fc = nn.Linear(dataset.num_node_features, hidden_channels)

        for _ in range(self.num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.out_fc = nn.Linear(hidden_channels, dataset.num_classes)

        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, x, adj_t):
        x = self.input_fc(x)
        x_input = x  # .copy()

        layer_out = []
        for i in range(self.num_layers):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)

            if i == 0:
                x = x + 0.2 * x_input
            else:
                x = x + 0.2 * x_input + 0.7 * layer_out[i - 1]
            layer_out.append(x)

        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]

        x = sum(layer_out)
        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_flag(model, data, train_idx, optimizer, device, args):

    y = data.y.squeeze(1)[train_idx]
    forward = lambda perturb : model(data.x+perturb, data.adj_t)[train_idx]
    model_forward = (model, forward)

    loss, _ = flag(model_forward, data.x.shape, y, args, optimizer, device, F.nll_loss)

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
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

    return train_acc, valid_acc, test_acc


def main():

    parser = argparse.ArgumentParser(description='OGBN-Arxiv (SAGE)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start-seed', type=int, default=0)

    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--attack', type=str, default='flag')

    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)


    model = SAGE_res(dataset, data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')


    vals, tests = [], []
    for run in range(args.runs):
        best_val, final_test = 0, 0

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):

            loss = train_flag(model, data, train_idx, optimizer, device, args)

            if epoch > args.epochs / 2 and epoch % args.test_freq == 0 or epoch == args.epochs:
                result = test(model, data, split_idx, evaluator)
                _, val, tst = result
                if val > best_val :
                    best_val = val
                    final_test = tst

        print(f'Run{run} val:{best_val}, test:{final_test}')
        vals.append(best_val)
        tests.append(final_test)

    print('')
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")


if __name__ == "__main__":
    main()