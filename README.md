# #55-DAG
This is a hasty readme for review only :P

We use three datasets: `'isAcyclic'`, `'MUTAG'` and `'highschool'`, change [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L52) (line 52 in main.py) to test them.

For _isAcyclic-n*_, you can set the number of nodes in the candidate [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L53) (line 53 in main.py), or input `'all'` to test for isAcyclic using the complete candidate space.

For isAcyclic and MUTAG dataset, you can set the GNN model to be `'GIN'` or `'GCN'` [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L54) (line 54 in main.py). For highschool dataset, it is GINE model by default.

The target class to be explained can be set [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L55) (line 55 in main.py).

Test the algorithm for one run: see [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L82-L89) (line 92-89 in main.py).

Alternatively, you can also repeat running the algorithm for multiple times by uncommenting [these](https://github.com/Gori-LV/DAG/blob/main/main.py#L91-L100) (line 91-100 in main.py).

The results and evaluation will be automatically printed and saved to the result folder. If you want to visualize the outputs, uncomment [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L89) (line 89 in main.py) or [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L100) (line 100 in main.py). WARNING: all outputs will be replaced.

