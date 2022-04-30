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

import torch_geometric
# import auxiliarymethods.datasets as dp
# from auxiliarymethods.gnn_evaluation import gnn_evaluation
# from gnn_baselines.gnn_architectures import GINE
# from tu_gnn_baselines.original_gnn_architectures import GINE0
# from tu_gnn_baselines.gnn_architectures import GINE0
import networkx as nx
from pytorchtools import EarlyStopping
from models import GINE0
# from gnn_baselines.gnn_architectures import GINEConv
# One training epoch for GNN model.
# def train(train_loader, model, optimizer, device):
#     model.train()
#     loss_out = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, data.y)
#         loss_out+=loss.item()
#         # print(loss.item())
#         loss.backward()
#         optimizer.step()
#
#     return loss_out / len(train_loader.dataset)


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
# dataset = 'sampled_subgraph_s5_l3_u7'

# print('working on '+dataset)
# train_set = TUDataset('data', name='highschool_ct2',use_edge_attr = True)
train_set = TUDataset('data/', name='highschool',use_edge_attr = True)

# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_edge_features, num_node_features, num_classes, num_layers, hidden = 1, 2, 2, 3, 32
model = GINE0(num_edge_features, num_node_features, num_classes, num_layers, hidden)
#
model.reset_parameters()
# model.load_state_dict(torch.load('data/backup_highschool/GINE0_l3_h32_b64_lr1_tr20.pt'))
model.load_state_dict(torch.load('checkpoints/highschool/highschool_GINE_best.pt'))

test_loader = DataLoader(train_set, batch_size=180, shuffle=False)
print(test(test_loader, model, device)) # for training acc on the original dataset


# ### for scores on candidates
# sample_data = TUDataset('result/'+path, dataset, use_edge_attr = True)
# sample_data = TUDataset('data/', 'sampled_subgraph_s5_l3_u7', use_edge_attr = True)
# graphs = read_graph('data/highschool/gSpan_output')
#
# data_items = []
# base = [0 for i in range(num_node_features)]
# size = {}
# for i in range(len(graphs)):
#     size[i] = {}
#     size[i]['n'] = graphs[i].number_of_nodes()
#     size[i]['e'] = graphs[i].number_of_edges()
#     data = torch_geometric.utils.convert.from_networkx(graphs[i])
#     x = []
#     for node in graphs[i].nodes:
#         tmp = base.copy()
#         tmp[graphs[i].nodes[node]['label']] = 1  # one-hot encoding
#         x.append(tmp)
#     data.x = torch.tensor(x, dtype=torch.float32)
#     edge_attr = []
#     for edge in graphs[i].edges:
#         edge_attr.append([graphs[i].edges[edge]['label']])
#         edge_attr.append([graphs[i].edges[edge]['label']])
#     data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
#     data_items.append(data)
#
# # score_loader = DataLoader(data_items, batch_size=len(data_items), shuffle=False)
# score_loader = DataLoader(sample_data, batch_size=len(sample_data), shuffle=False)
# #
# data, slices = read_tu_data('result/highschool/gSpan_output_data/raw','sampled_subgraph_s5_l3_u7')
# batch = read_file('result/highschool/gSpan_output_data/raw', 'sampled_subgraph_s5_l3_u7', 'graph_indicator', dtype=torch.long)-1
# test_data, slices = split(data, batch)
# score_loader = DataLoader(data, batch_size=31495,shuffle=False)
# model.eval()
# with torch.no_grad():
#     for d in score_loader:
#         output=model(d)
# score = [[x.item() for x in nn.Softmax(dim=0)(o)] for o in output]
# print(score[:5])