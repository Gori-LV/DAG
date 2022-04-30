import networkx as nx
import networkx.algorithms.isomorphism as iso

import torch_geometric
import matplotlib.pyplot as plt
import torch

def genSyn():
	datasets = []
	labels = []
	for i in range(2,9):
		for j in range(2,9):
			datasets.append(nx.grid_2d_graph(i, j))
			labels.append(0)
	for i in range(3,65):
		datasets.append(nx.cycle_graph(i))
		labels.append(0)

	for i in range(20):
		datasets.append(nx.cycle_graph(3))
		labels.append(0)


	for i in range(2,65):
		datasets.append(nx.wheel_graph(i))
		labels.append(0)

	for i in range(2,35):
		datasets.append(nx.circular_ladder_graph(i))
		labels.append(0)





	for i in range(2,65):
		datasets.append(nx.star_graph(i))
		labels.append(1)


	g= nx.balanced_tree(2, 5)
	datasets.append(g)
	labels.append(1)
	for i in range(62, 2, -1):
		g.remove_node(i)
		datasets.append(g)
		labels.append(1)

	for i in range(3,65):
		datasets.append(nx.path_graph(i))
		labels.append(1)


	for i in range(3,5):
		for j in range(5,65):
			datasets.append(nx.full_rary_tree(i,j))
			labels.append(1)

	data_items = []
	for i in range(len(datasets)):
		data = torch_geometric.utils.convert.from_networkx(datasets[i])
		# base=[0 for i in range(64)]
		# x=[]
		# for node in datasets[i].degree:
		# 	tmp = base.copy()
		# 	tmp[datasets[i].degree[node[0]]-1]=1
		# 	x.append(tmp)
		base=[0 for i in range(3)]
		x=[]
		for node in datasets[i].degree:
			tmp = base.copy()
			if datasets[i].degree[node[0]]<20:
				tmp[0]=1
			elif 20<=datasets[i].degree[node[0]]<40:
				tmp[1]=1
			else:
				tmp[-1]=1
			x.append(tmp)

		data.x = torch.tensor(x, dtype=torch.float32)
		data.y = torch.tensor([labels[i]],dtype=torch.long)
		print(labels[i])
		data_items.append(data)
	return datasets, data_items

datasets, data_items = genSyn()

cyclic_item = [datasets[i] for i in range(len(data_items)) if data_items[i].y.item()==0] # label 0 is cyclic
print(len(cyclic_item))
acyclic_item = [datasets[i] for i in range(len(data_items)) if data_items[i].y.item()==1] # label 0 is cyclic
print(len(acyclic_item))

### XGNN result for cyclic class
XGNN_cyclic = []
G = nx.Graph() # 3 node
G.add_edges_from([(0,1),(1,2),(0,2)])
XGNN_cyclic.append(G)

G = nx.Graph() # 4 node
G.add_edges_from([(0,1),(1,2),(0,2),(0,3),(1,2),(1,3),(2,3)])
XGNN_cyclic.append(G)

G = nx.Graph() # 5 node
G.add_edges_from([(0,1),(1,2),(0,2),(0,3),(1,2),(1,3),(2,3),(0,5),(1,5)])
XGNN_cyclic.append(G)

G = nx.Graph() # 6 node
G.add_edges_from([(0,1),(1,2),(0,2),(0,3),(1,2),(1,3),(2,3),(0,5),(1,5),(1,6),(2,6),(5,6)])
XGNN_cyclic.append(G)

G = nx.Graph() # 7 node
G.add_edges_from([(0,1),(1,2),(0,2),(0,3),(1,2),(1,3),(2,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5),(4,5)])
XGNN_cyclic.append(G)
# nx.draw(XGNN_cyclic[4])

### XGNN result of acyclic class
XGNN_acyclic = []
G = nx.Graph() # 3 node
G.add_edges_from([(0,1),(1,2)])
XGNN_acyclic.append(G)

G = nx.Graph() # 4 node
G.add_edges_from([(0,1),(1,2),(1,3)])
XGNN_acyclic.append(G)

G = nx.Graph() # 5 node
G.add_edges_from([(0,1),(1,2),(1,3),(1,4)])
XGNN_acyclic.append(G)

G = nx.Graph() # 6 node
G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5)])
XGNN_acyclic.append(G)

G = nx.Graph() # 7 node
G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(4,6)])
XGNN_acyclic.append(G)
# nx.draw(XGNN_acyclic[0])
### check self-sup for XGNN

def metric(pattern):
	c_count = 0
	for g in cyclic_item:
		GM = iso.GraphMatcher(g, pattern)
		if GM.subgraph_is_isomorphic():
			c_count+=1
	# print(c_count)
	a_count = 0
	for g in acyclic_item:
		GM = iso.GraphMatcher(g, pattern)
		if GM.subgraph_is_isomorphic():
			a_count+=1
	# print(a_count)
	return c_count, a_count

### check cyclic class sup
mean_sup=0
c=0
for p in XGNN_cyclic:
	sup, den = metric(p)
	if sup+den!=0:
		mean_sup+=sup/(sup+den)
		c+=1
		print(sup/(sup+den))
	else:
		print('OOD')
print(mean_sup/c)
### check acyclic class sup
mean_sup=0
c=0
for p in XGNN_acyclic:
	den, sup = metric(p)
	if sup+den!=0:
		mean_sup+=sup/(sup+den)
		c+=1
		print(sup/(sup+den))
	else:
		print('OOD')
print(mean_sup/c)
# output:
# 1.0
# 1.0
# OOD
# OOD
# OOD
# final: 1.0
# 0.6011787819253438
# 0.5705329153605015
# 0.6605166051660517
# 0.44476744186046513
# 0.562874251497006
# final: 0.5679739991618737




# ###
# content = ''
# for i in range(len(data_items)):
# 	content+='t # '+str(i)+'\n'
# 	# nodes = data_items[i].edge_index[0].tolist()
# 	# nodes = list(set(nodes))
# 	# for node in nodes:
# 	# 	content+= 'v '+str(node)+ ' 0\n'
# 	for j in range(data_items[i].x.shape[0]):
# 		content+='v '+ str(j)+' '+str(data_items[i].x[j].nonzero().item())+'\n'
# 	edges_tmp = [[data_items[i].edge_index[0][x].item(), data_items[i].edge_index[1][x].item()] for x in range(data_items[i].edge_index.shape[1])]
# 	edges = set()
# 	for edge in edges_tmp:
# 		edge.sort()
# 		edges.add(tuple(edge))
# 	for edge in edges:
# 		content +='e '+str(edge[0])+' '+str(edge[1])+ ' 0\n'
# content+='t # -1'
# with open('isAcyclic_rich','w') as f:
# 	f.write(content)
# ###


# ### code for showing raw figures
# savefig_path = 'datasets/isAcyclic/raw_figure/'
# c=0
# for gnx in datasets:
#     # plt.figure(1, figsize=fsize)
#     plt.figure(c)
#     # plt.box(False)
#     # pos = nx.kamada_kawai_layout(gnx)
#     # nx.draw_networkx(gnx, pos, arrows=True, with_labels=True, labels=vlbs)
#     # nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)
#     nx.draw(gnx)
#     plt.savefig(savefig_path+ str(c)+'.png',bbox_inches="tight")
#     # plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',transparent=True)
#     plt.close(c)
#     c+=1
# ###
