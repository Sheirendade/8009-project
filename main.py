from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
print(dataset)

data = dataset[0]
print(data)

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
print(dataset)

data = dataset[0]
print(data)

split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')

train_idx = split_idx['train']
val_idx = split_idx['valid']
test_idx = split_idx['test']
print(train_idx.shape)
print(val_idx.shape)
print(test_idx.shape)

y = data.y.squeeze(1)[train_idx]
print(y.shape)

from torch_geometric.loader import NeighborSampler
train_loader = NeighborSampler(edge_index=data.adj_t, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024, shuffle=True, num_workers=12)
print(train_loader)