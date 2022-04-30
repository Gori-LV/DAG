# DAG
This is a hasty readme for review only :stuck_out_tongue:

We use three datasets: `'isAcyclic'`, `'MUTAG'` and `'highschool'`, change [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L52) to test them.

For _isAcyclic-n*_, you can set the number of nodes in the candidate [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L53), or input `'all'` to test for isAcyclic using the complete candidate space.

For isAcyclic and MUTAG dataset, you can set the GNN model to be `'GIN'` or `'GCN'` [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L54). For highschool dataset, it is GINE model by default.

The target class to be explained can be set [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L55).

Test the algorithm for one run: see [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L79-L85).

Alternatively, you can also repeat running the algorithm for multiple times by uncommenting [these](https://github.com/Gori-LV/DAG/blob/main/main.py#L87-L94).

The results and evaluation will be automatically printed and saved to the result folder. If you want to visualize the outputs, uncomment [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L85) or [here](https://github.com/Gori-LV/DAG/blob/main/main.py#L94) (WARNING: all outputs will be replaced).

