import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import DataLoader
from models import GINE0
import os
import torch
import shutil
import numpy as np
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
import networkx as nx

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


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest.pt"
    best_pth_name = f'{model_name}_best.pt'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))

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

# path = "highschool_ct2/"
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
# model.load_state_dict(torch.load('checkpoints/highschool/highschool_GINE_best.pt'))

train_loader = DataLoader(train_set, batch_size=180, shuffle=True)

print('start training model==================')
optimizer = Adam(model.parameters(), lr=0.0005)

best_acc = 0.0
best_loss = -100.0
data_size = len(train_set)
print(f'The total num of dataset is {data_size}')

# save path for model
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.isdir(os.path.join('checkpoints', 'highschool')):
    os.mkdir(os.path.join('checkpoints', f"{'highschool'}"))
ckpt_dir = "checkpoints/highschool/"


## for training

criterion = nn.CrossEntropyLoss()

early_stop_count = 0
for epoch in range(200):
    acc = []
    loss_list = []
    model.train()
    for batch in train_loader:
        # logits, probs, _ = model(data=batch)
        logits = model(data=batch)
        loss = criterion(logits, batch.y)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        optimizer.step()

        ## record
        _, prediction = torch.max(logits, -1)
        loss_list.append(loss.item())
        acc.append(prediction.eq(batch.y).cpu().numpy())

    # report train msg
    epoch_acc = np.concatenate(acc, axis=0).mean()
    epoch_loss = np.average(loss_list)
    print(f"Train Epoch:{epoch}  |Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.3f}")

    # only save the best model
    is_best = (epoch_acc > best_acc) or (epoch_loss < best_loss and epoch_acc >= best_acc)
    # if epoch_acc == best_acc:
    #     early_stop_count += 1
    # # if early_stop_count > train_args.early_stopping:
    # if early_stop_count > 20:
    #     break
    if is_best:
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # early_stop_count = 0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
    if is_best or epoch % 50 == 0:
        save_best(ckpt_dir, epoch, model, 'highschool_GINE', epoch_acc, is_best)

print(f"The best training accuracy is {best_acc}.")



test_loader = DataLoader(train_set, batch_size=180, shuffle=False)
#
model.reset_parameters()
# model.load_state_dict(torch.load('data/backup_highschool/GINE0_l3_h32_b64_lr1_tr20.pt'))
model.load_state_dict(torch.load('checkpoints/highschool/highschool_GINE_best.pt')['net'])

print(test(test_loader, model, device)) # for training acc on the originEal dataset

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