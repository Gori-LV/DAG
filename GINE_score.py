import matplotlib.pyplot as plt

from random import sample
import json
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.transforms as T
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
# from torch_geometric.utils import degree

from torch_geometric.utils import from_networkx
# import auxiliarymethods.datasets as dp
# from auxiliarymethods.gnn_evaluation import gnn_evaluation
# from gnn_baselines.gnn_architectures import GINE
# from tu_gnn_baselines.original_gnn_architectures import GINE, GINEWithJK, GINE0
from models import GINE0
import networkx as nx

# from models import GINE0
from pytorchtools import EarlyStopping
# from gnn_baselines.gnn_architectures import OneHotEdge

# from gnn_baselines.gnn_architectures import GINEConv

# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()
    loss_out = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss_out+=loss.item()
        # print(loss.item())
        loss.backward()
        optimizer.step()

    return loss_out / len(train_loader.dataset)


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    loss_out = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss_out+=loss.item()
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset), loss_out / len(loader.dataset)


def read_graph(file):
    with open(file, 'r') as f:
        content = f.readlines()
    graphs = []
    for line in content[:-1]:
        tmp = line.strip().split()
        if len(tmp)==0:
            continue
        elif tmp[0]=='t':
            graphs.append(nx.Graph(id=int(tmp[-1])))
        elif tmp[0]== 'v':
            graphs[-1].add_nodes_from([(int(tmp[1]), {"label": int(tmp[-1])})])
        elif tmp[0]=='e':
            graphs[-1].add_edges_from([(int(tmp[1]),int(tmp[2]), {"label": int(tmp[-1])})])
        else:
            continue
    return graphs

path = "highschool_ct2/"
dataset = 'sampled_subgraph_s5_v12'

# print('working on '+dataset)
train_set = TUDataset('data', name='highschool_ct2',use_edge_attr = True)

# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_num_epochs=500
batch_size=64

l = 3 #layer
h = 32 # hidden
model = GINE0(train_set, l, h).to(device)
model.reset_parameters()
model.load_state_dict(torch.load('data/highschool_ct2/GINE0_l3_h32_b64_lr1_tr20.pt'))

sample_data = TUDataset('result/'+path, dataset, use_edge_attr = True)

score_loader = DataLoader(sample_data, batch_size=len(sample_data), shuffle=False)
for d in score_loader:
    # output = model(d)
    output=model(d).data
score = []
for o in output:
    score.append([x.item() for x in nn.Softmax(dim=0)(o)])

print(max([x[0] for x in score]))
print(max([x[1] for x in score]))

# for re-running the code and avoid changing existing file
# with open('result/'+path+dataset+'/GNN_score.json', 'w') as f:
#     json.dump(score, f, indent=4)
# print('GNN score saved')

# # to test average acc
# n_rep = 1000
# acc = 0
# test_ratio = .2
# for i in range(n_rep):
#     dataset.shuffle()
#     test_index = sample(list(range(len(dataset))), int(test_ratio*len(dataset)))
#     train_index = [x for x in list(range(len(dataset))) if x not in test_index]
#
#     # Sample 10% split from training split for validation.
#     train_index, val_index = train_test_split(train_index, test_size=test_ratio)
#
#     # Split data.
#     train_dataset = dataset[train_index]
#     val_dataset = dataset[val_index]
#     test_dataset = dataset[test_index]
#
#     # Prepare batching.
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
#     acc += test(test_loader, model, device)[0]
#     i+=1
#     print(i)
# print('avg acc is : ' +str(acc/n_rep)) #n = 1000 avg acc is : 0.978249999999993
