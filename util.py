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
    # vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
    # elbs = {}
    # for vid, v in self.vertices.items():
    #     gnx.add_node(vid, label=v.vlb)
    # for vid, v in self.vertices.items():
    #     for to, e in v.edges.items():
    #         if (not self.is_undirected) or vid < to:
    #             gnx.add_edge(vid, to, label=e.elb)
    #             elbs[(vid, to)] = e.elb
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
    nx.draw_networkx_nodes(gnx, pos, nodelist=red_node, node_color="darkred",node_size=250)
    nx.draw_networkx_nodes(gnx, pos, nodelist=green_node, node_color="g",node_size=250)
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
        width=18,
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
    # vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
    # elbs = {}
    # for vid, v in self.vertices.items():
    #     gnx.add_node(vid, label=v.vlb)
    # for vid, v in self.vertices.items():
    #     for to, e in v.edges.items():
    #         if (not self.is_undirected) or vid < to:
    #             gnx.add_edge(vid, to, label=e.elb)
    #             elbs[(vid, to)] = e.elb
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
    # vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
    # elbs = {}
    # for vid, v in self.vertices.items():
    #     gnx.add_node(vid, label=v.vlb)
    # for vid, v in self.vertices.items():
    #     for to, e in v.edges.items():
    #         if (not self.is_undirected) or vid < to:
    #             gnx.add_edge(vid, to, label=e.elb)
    #             elbs[(vid, to)] = e.elb
    fsize = (min(16, 1 * len(v)),
             min(16, 1 * len(v)))

    plt.figure(gnx.graph['id'], figsize=fsize)
    plt.box(False)
    pos = nx.kamada_kawai_layout(gnx)
    nx.draw_networkx(gnx, pos, arrows=False, node_color = 'dodgerblue', with_labels=False)
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

if __name__ == '__main__':
    #
    # # for test, explain_result from MUTAG_non_mtg_no_edge_gSpan_output_s4_l3_explain_result
    # # explanantion for class 0
    # # exp_0 = [(8589, 0), (27236, 0), (2, 0)]
    # exp_0 = [(158, 0), (216, 0), (52, 0), (856, 0)]
    # # explanantion for class 1
    # # exp_1 = [(1548, 1), (839, 1), (2086, 1), (15, 1), (1584, 1), (2742, 1), (2687, 1), (815, 1), (825, 1), (1386, 1), (670, 1), (2587, 1), (1192, 1), (995, 1), (591, 1), (2160, 1), (2045, 1), (1838, 1), (2649, 1), (2442, 1), (906, 1), (824, 1), (89, 1), (758, 1), (338, 1), (850, 1), (2560, 1), (341, 1), (853, 1), (2192, 1), (590, 1), (2615, 1), (997, 1), (1601, 1), (2559, 1), (1735, 1), (1538, 1), (2565, 1), (2401, 1), (1288, 1), (497, 1), (1317, 1), (4, 1), (2285, 1), (1435, 1), (519, 1), (499, 1), (1523, 1), (781, 1), (410, 1), (1749, 1), (1460, 1), (2185, 1), (1187, 1), (1948, 1), (44, 1), (2473, 1), (2581, 1), (963, 1), (1186, 1), (2187, 1), (414, 1), (1225, 1), (1333, 1), (2186, 1), (2583, 1), (1700, 1), (1188, 1), (768, 1), (141, 1), (2159, 1), (1450, 1), (1397, 1), (2083, 1), (599, 1), (1846, 1), (1311, 1), (1882, 1), (1678, 1), (37, 1), (2308, 1), (1310, 1), (2741, 1), (1966, 1), (774, 1), (2179, 1), (2691, 1), (1585, 1), (98, 1), (1181, 1), (718, 1), (2172, 1), (2175, 1)]
    # # exp_1 = [(31021, 1), (12126, 1), (9576, 1), (11246, 1), (9687, 1), (21125, 1), (9004, 1), (31020, 1), (10005, 1), (8981, 1), (18338, 1), (31965, 1), (20924, 1), (10067, 1), (9151, 1), (9755, 1), (8731, 1), (25223, 1), (27281, 1), (31823, 1), (8901, 1), (12511, 1), (12222, 1), (13850, 1), (10069, 1), (22580, 1), (10896, 1), (21559, 1), (14936, 1), (12054, 1), (23840, 1), (11437, 1), (15963, 1), (9993, 1), (23843, 1), (24578, 1), (23869, 1), (27282, 1), (19405, 1), (22001, 1), (19211, 1), (12118, 1), (13549, 1), (23188, 1), (8209, 1), (26063, 1), (28715, 1), (12042, 1), (24727, 1), (12186, 1), (28035, 1), (9652, 1), (5881, 1), (23397, 1), (9888, 1), (23820, 1), (17545, 1), (11450, 1), (9799, 1), (29422, 1), (8985, 1), (28575, 1), (28148, 1), (26100, 1), (13904, 1), (10119, 1), (28075, 1), (5796, 1), (15839, 1), (28219, 1), (28839, 1), (8789, 1), (20732, 1), (10892, 1), (11207, 1), (9359, 1), (5808, 1), (11919, 1), (26829, 1), (24781, 1), (11206, 1), (18804, 1), (8334, 1), (11918, 1), (25066, 1), (24728, 1), (11393, 1), (11419, 1), (16591, 1), (8110, 1), (9564, 1), (15934, 1), (9577, 1), (31593, 1), (28235, 1), (17683, 1), (22852, 1)]
    # exp_1 = [(365, 1), (715, 1), (534, 1), (156, 1), (301, 1), (845, 1), (636, 1), (205, 1), (177, 1), (262, 1), (275, 1), (660, 1), (685, 1), (291, 1), (155, 1), (707, 1), (514, 1), (500, 1), (635, 1), (617, 1), (495, 1), (384, 1), (369, 1), (632, 1), (247, 1), (598, 1), (756, 1), (490, 1), (268, 1), (347, 1), (106, 1), (509, 1), (23, 1), (528, 1), (526, 1), (114, 1), (424, 1), (593, 1), (148, 1), (693, 1), (42, 1), (403, 1), (300, 1), (210, 1), (109, 1), (545, 1), (570, 1), (405, 1), (639, 1), (98, 1), (137, 1), (358, 1), (111, 1), (407, 1), (858, 1), (371, 1), (746, 1), (547, 1), (835, 1), (614, 1), (778, 1), (258, 1), (376, 1), (81, 1), (792, 1), (572, 1), (302, 1), (744, 1), (0, 1), (26, 1), (479, 1), (589, 1), (437, 1), (402, 1), (312, 1), (282, 1), (556, 1), (831, 1), (748, 1), (142, 1), (496, 1), (422, 1), (397, 1), (481, 1), (562, 1), (395, 1), (24, 1), (848, 1), (176, 1), (482, 1), (77, 1), (558, 1), (550, 1), (770, 1), (729, 1), (764, 1), (515, 1), (710, 1), (651, 1), (595, 1), (759, 1), (10, 1), (613, 1), (861, 1), (686, 1), (806, 1), (303, 1), (573, 1), (394, 1), (175, 1), (120, 1), (587, 1), (762, 1), (813, 1), (314, 1), (539, 1), (416, 1), (533, 1), (623, 1), (805, 1), (306, 1), (620, 1), (492, 1), (274, 1), (546, 1), (521, 1), (141, 1), (337, 1), (445, 1), (457, 1), (638, 1), (603, 1), (862, 1), (107, 1), (684, 1), (776, 1), (747, 1), (408, 1), (696, 1), (317, 1), (468, 1), (722, 1), (290, 1), (797, 1), (449, 1), (711, 1), (824, 1), (665, 1), (230, 1), (543, 1), (446, 1), (348, 1), (430, 1), (501, 1), (115, 1), (548, 1), (475, 1), (418, 1), (259, 1), (607, 1), (811, 1), (774, 1), (328, 1), (398, 1), (502, 1), (718, 1), (814, 1), (334, 1), (541, 1), (140, 1), (83, 1), (676, 1), (578, 1), (113, 1), (255, 1), (332, 1), (221, 1), (622, 1), (739, 1), (220, 1), (360, 1), (659, 1), (273, 1), (582, 1), (497, 1), (356, 1), (628, 1), (308, 1), (224, 1), (626, 1), (641, 1), (600, 1), (382, 1), (124, 1), (289, 1), (389, 1), (656, 1), (212, 1), (681, 1), (557, 1), (45, 1), (478, 1), (596, 1), (719, 1), (340, 1), (559, 1), (207, 1), (517, 1), (67, 1), (336, 1), (295, 1), (393, 1), (713, 1), (364, 1), (809, 1), (630, 1), (624, 1), (231, 1), (703, 1), (392, 1), (381, 1), (705, 1), (606, 1), (670, 1), (133, 1), (773, 1), (634, 1), (612, 1), (307, 1), (560, 1), (396, 1), (750, 1), (438, 1), (709, 1), (488, 1), (682, 1), (765, 1), (602, 1), (463, 1), (619, 1), (749, 1), (333, 1), (329, 1), (512, 1), (576, 1), (629, 1), (401, 1), (819, 1), (444, 1), (826, 1), (524, 1), (35, 1), (691, 1), (625, 1), (766, 1), (355, 1), (687, 1), (362, 1), (361, 1), (256, 1), (415, 1), (277, 1), (833, 1), (621, 1), (191, 1), (414, 1), (816, 1), (421, 1), (633, 1), (796, 1), (406, 1), (810, 1), (241, 1), (779, 1), (527, 1), (44, 1), (309, 1), (697, 1), (847, 1), (520, 1), (531, 1), (476, 1), (518, 1), (100, 1), (537, 1), (808, 1), (780, 1), (565, 1), (493, 1), (694, 1), (592, 1), (637, 1)]
    #
    # input_path = './gnn-model-explainer-master/gSpan_ouput/'
    # file = 'MUTAG_data_no_edge_s9_l4_u9_gSpan'
    # savefig_path = 'explain_result/'+file+'_result/'
    # output = exp_0+exp_1
    #
    # visualResult(output, input_path, file, savefig_path)

    gSpanOutput = 'result/highschool_ct2/sampled_subgraph_s5_l3_u7/gSpan_output'
    output = [(2713, 1)]
    visualResult(output, gSpanOutput, if_highschool=True)
