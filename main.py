import math
import statistics
# from gSpan import candidate_generation
from util import visualResult
# from model_score import getScore
import argparse
import os
import torch
import torch.nn as nn
# import os
import json
# import re
import numpy as np
# import networkx as nx
# import explain
from models import *
from explain import *
# from tu_gnn_baselines.original_gnn_architectures import GINE0

import time

def loadGNN(dataset, gnn_name):
	ckpt_dir = 'checkpoints/'+dataset+'/'

	if dataset == 'highschool':
		gnn_name='GINE'
		num_edge_features, num_node_features, num_classes, num_layers, hidden = 1, 2, 2, 3, 32
		model = GINE0(num_edge_features, num_node_features, num_classes, num_layers, hidden)
		model.reset_parameters()
		ckpt_dir = 'checkpoints/'+dataset+'/'
		saved_net = 'highschool_GINE_best.pt'
		model.load_state_dict(torch.load(ckpt_dir+saved_net)['net'])
		print(gnn_name +' model for ' +dataset + ' is loaded.')
		return model

	elif dataset == 'MUTAG':
		if gnn_name=='GIN':
			model = GIN(model_level='graph', dim_node=7, dim_hidden=64, num_classes=2,num_layer=3)			
		elif gnn_name=='GCN':
			model = GCN(model_level='graph', dim_node=7, dim_hidden=[128, 128, 128], ffn_dim=[64],
						num_classes=2)
	elif dataset == 'isAcyclic':
		if gnn_name=='GIN':
			model = GIN(model_level='graph', dim_node=3, dim_hidden=64, num_classes=2, num_layer=3)
		elif gnn_name=='GCN':
			model = GCN(model_level='graph', dim_node=3, dim_hidden=[8, 16], ffn_dim= [32], num_classes=2)
	saved_net = dataset+'_'+gnn_name+'_best.pt'
	checkpoint = torch.load(ckpt_dir+saved_net)
	model.load_state_dict(checkpoint['net'])
	print(gnn_name +' model for ' +dataset + ' is loaded.')
	return model


if __name__ == '__main__':

	with open('parameters.json','r') as f:
		parameters = json.load(f)

	dataset = 'MUTAG'
	n_nodes = 'all' # for isAcyclic dataset only
	gnn = 'GIN' # for MUTAG and isAcyclic datasets only
	target_class = 1
	model = loadGNN(dataset, gnn)

	if_highschool = False
	if_isAcyclic = False

	# Overall fidelity, total support, average denial, size penalty
	if dataset=='isAcyclic':
		if_isAcyclic = True
		if n_nodes=='all':
			lambdas = np.array(parameters[dataset][gnn][str(7)], dtype=np.int64)
		else:
			lambdas = np.array(parameters[dataset][gnn][str(n_nodes)], dtype=np.int64)
		explainer = DAG(dataset=dataset, model=model, lambdas=lambdas, isAcyclic_n_nodes=n_nodes)
	else:
		if dataset=='highschool':
			if_highschool=True
			gnn ='GINE'
		lambdas = np.array(parameters[dataset][gnn], dtype=np.int64)
		explainer = DAG(dataset = dataset, model=model, lambdas=lambdas)

	n_candidate = len([x for x in explainer.candidate if x[1]==target_class])
	k = math.ceil((n_candidate+1)/2)

	##### For one time test
	start = time.time()
	output = explainer.explain(k=k, target_class=target_class)
	end = time.time()
	print('time used')
	print(end - start)

	print('\nEvaluation on the output set:')
	explainer.evalOutput(output, read_out=True)
	# visualResult(output=output, gSpanOutput=explainer.gSpan_output_file, if_highschool=if_highschool, if_isAcyclic=if_isAcyclic, save_path=explainer.save_path)

	# #### for repeating test
	# repeat = 10
	# start = time.time()
	# _, output = explainer.repeatExplain(k, repeat, target_class=target_class, save=True)
	# end = time.time()
	# print('time used')
	# print((end - start)/repeat)
	# #
	# # print('final output')
	# # print(output)
	# print('\nEvaluation on final explanation set:')
	# explainer.evalOutput(output, read_out=True)
	# # visualResult(output=output, gSpanOutput=explainer.gSpan_output_file, if_isAcyclic=if_isAcyclic, if_highschool= if_highschool, save_path=explainer.save_path)
	# # print(explainer.calObj(output)/explainer.Lambda[0])
