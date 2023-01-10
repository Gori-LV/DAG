import networkx as nx
import matplotlib.pyplot as plt
import re
import os
import shutil

def plot_highschool(chunk, savefig_path):

    graph = chunk.strip().split('\n')
    # print('working on pattern # '+graph[0])
    v = [[int(y) for y in x.split(' ')[1:]] for x in graph if x[0]=='v']
    e = [[int(y) for y in x.split(' ')[1:]] for x in graph if x[0]=='e']

    gnx = nx.Graph(id=graph[0])

    gnx.add_nodes_from([(x[0], {'label': x[1]}) for x in v])
    gnx.add_edges_from([(x[0],x[1], {'label':x[-1]}) for x in e])

    vlbs = {x[0]:x[1] for x in v}
    elbs = {(x[0],x[1]):x[-1] for x in e}

    fsize = (min(16, 1 * len(v)),
             min(16, 1 * len(v)))

    plt.figure(gnx.graph['id'], figsize=fsize)
    plt.box(False)
    pos = nx.spring_layout(gnx)

    red_node = [x for x in gnx.nodes if gnx.nodes[x]['label']==1]
    green_node = [x for x in gnx.nodes if gnx.nodes[x]['label']==0]

    green_edge = [x for x in gnx.edges if gnx.edges[x]['label']==0]
    red_edge = [x for x in gnx.edges if gnx.edges[x]['label']==1]
    darkred_edge = [x for x in gnx.edges if gnx.edges[x]['label']==2]
    maroon_edge = [x for x in gnx.edges if gnx.edges[x]['label']==3]
    dark_edge = [x for x in gnx.edges if gnx.edges[x]['label']==4]
    bunt_edge = [x for x in gnx.edges if gnx.edges[x]['label']>4]
    # print(green_node)
    propagation = [x for x in gnx.edges if gnx.nodes[x[0]]['label']+gnx.nodes[x[1]]['label']==2 and gnx.edges[x]['label']!=0]

    # nx.draw_networkx_nodes(gnx, pos, nodelist=red_node, node_color="darkred",node_size=0)
    # nx.draw_networkx_nodes(gnx, pos, nodelist=green_node, node_color="g",node_size=0)
    nx.draw_networkx_nodes(gnx, pos, nodelist=red_node, node_color="darkred",node_size=350)
    nx.draw_networkx_nodes(gnx, pos, nodelist=green_node, node_color="g",node_size=350)
    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=red_edge,
        width=4,
        alpha=.7,
        edge_color="red",
    )

    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=darkred_edge,
        width=4,
        alpha=.7,
        edge_color="darkred",
    )

    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=maroon_edge,
        width=4,
        alpha=.6,
        edge_color="maroon",
    )

    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=dark_edge,
        width=4,
        alpha=.8,
        edge_color="maroon",
    )

    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=bunt_edge,
        width=4,
        alpha=1,
        edge_color="maroon",
    )

    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=green_edge,
        width=4,
        alpha=.5,
        edge_color="green",
    )

    nx.draw_networkx_edges(
        gnx,
        pos,
        edgelist=propagation,
        width=20,
        alpha=.3,
        edge_color="red",
    )

    # plt.title('s = ', y=-0.01, fontsize = 60)
    # plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png', transparent=True)
    plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',bbox_inches="tight")
    plt.close(gnx.graph['id'])
    # plt.show()

def plot(chunk, savefig_path):

    graph = chunk.strip().split('\n')
    # print('working on pattern # '+graph[0])
    v = [[int(y) for y in x.split(' ')[1:]] for x in graph if x[0]=='v']
    e = [[int(y) for y in x.split(' ')[1:]] for x in graph if x[0]=='e']

    gnx = nx.Graph(id=graph[0])

    gnx.add_nodes_from([(x[0], {'label': x[1]}) for x in v])
    gnx.add_edges_from([(x[0],x[1], {'label':x[-1]}) for x in e])

    vlbs = {x[0]:x[1] for x in v}
    elbs = {(x[0],x[1]):x[-1] for x in e}

    fsize = (min(16, 1 * len(v)),
             min(16, 1 * len(v)))

    plt.figure(gnx.graph['id'], figsize=fsize)
    plt.box(False)
    pos = nx.kamada_kawai_layout(gnx)
    nx.draw_networkx(gnx, pos, arrows=True, with_labels=True, labels=vlbs)
    nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)

    plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',bbox_inches="tight")
    # plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',transparent=True)
    plt.close(gnx.graph['id'])
    # plt.show()

def plot_isAcyclic(chunk, savefig_path):

    graph = chunk.strip().split('\n')
    # print('working on pattern # '+graph[0])
    v = [[int(y) for y in x.split(' ')[1:]] for x in graph if x[0]=='v']
    e = [[int(y) for y in x.split(' ')[1:]] for x in graph if x[0]=='e']

    gnx = nx.Graph(id=graph[0])

    gnx.add_nodes_from([(x[0], {'label': x[1]}) for x in v])
    gnx.add_edges_from([(x[0],x[1], {'label':x[-1]}) for x in e])

    vlbs = {x[0]:x[1] for x in v}
    elbs = {(x[0],x[1]):x[-1] for x in e}

    fsize = (min(16, 1 * len(v)),
             min(16, 1 * len(v)))

    plt.figure(gnx.graph['id'], figsize=fsize)
    plt.box(False)
    pos = nx.kamada_kawai_layout(gnx)
    nx.draw_networkx(gnx, pos, arrows=False, node_color = 'dodgerblue', with_labels=False)
    # nx.draw_networkx(gnx, pos, arrows=False, node_color = 'orange', with_labels=False) # for glocal
    # nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)

    plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',bbox_inches="tight")
    # plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',transparent=True)
    plt.close(gnx.graph['id'])
    # plt.show()

def visualResult(output, gSpanOutput, save_path, if_highschool = False, if_isAcyclic = False):
    label = sorted(list(set([x[-1] for x in output])))
    # save_path = gSpanOutput.replace('gSpan_output', '')
    for l in label:
        if not os.path.exists(save_path+'class_'+str(l)):
            os.makedirs(save_path+'class_'+str(l))
        else:
            print(save_path+'class_'+str(l)+' already exists, delete and give new visual')
            # os.rmdir(save_path+'class_'+str(l))
            shutil.rmtree(save_path+'class_'+str(l))
            os.makedirs(save_path+'class_'+str(l))

    with open(gSpanOutput, 'r') as f:
        content = f.read()
    chunks = re.findall(r"t #(.*?)Support", content, flags=re.S)

    patterns = []
    patterns.append([x[0] for x in [y for y in output if y[-1]==0]])
    patterns.append([x[0] for x in [y for y in output if y[-1]==1]])

    for l in label:
        savefig_path = save_path + 'class_' + str(l)+'/'
        for item in [chunks[x] for x in patterns[l]]:
            if if_highschool:
                plot_highschool(item, savefig_path)
            elif if_isAcyclic:
                plot_isAcyclic(item, savefig_path)
            else:
                plot(item, savefig_path)

def diversityWeight(file):
    with open(file, 'r') as f:
        content = f.read()
    chunks = re.findall(r"t #(.*?)Support", content, flags=re.S)
    graphs_weight = {}
    for item in chunks:
        graph = item.strip().split('\n')
        # print('working on pattern # '+graph[0])
        v_labl = set([int(x.split(' ')[-1]) for x in graph if x[0]=='v'])
        e_labl = set([int(x.split(' ')[-1]) for x in graph if x[0]=='e'])
        graphs_weight[int(graph[0])]=len(v_labl)+len(e_labl)
    return graphs_weight
