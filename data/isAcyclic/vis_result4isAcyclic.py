import networkx as nx
import matplotlib.pyplot as plt

with open('isAcyclic_0_s2_l3_u7/final_result', 'r') as f:
    content = f.readlines()

graphs = []
for line in content:
    tmp = line.strip().split()
    if len(tmp)==0:
        continue
    elif tmp[0]=='Mined':
        graphs.append(nx.Graph(id=int(tmp[-1])))
    elif tmp[0]== 'v':
        graphs[-1].add_nodes_from([(int(tmp[1]), {"label": int(tmp[-1])})])
    elif tmp[0]=='e':
        graphs[-1].add_edges_from([(int(tmp[1]),int(tmp[2]), {"label": int(tmp[-1])})])
    elif tmp[0]=='Support':
        graphs[-1].graph['support']=int(tmp[-1])
    else:
        continue


for g in graphs:
    plt.figure()
    plt.box(False)
    nx.draw(g, edge_color = 'gray',node_color='dodgerblue')
    plt.savefig('isAcyclic_0_s2_l3_u7/final_vis/'+str(g.graph['id'])+'.pdf', bbox_inches="tight", transparent=True)
    plt.close()
