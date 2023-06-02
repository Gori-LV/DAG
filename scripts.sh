#!/bin/sh
# isAcyclic dataset

python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 0 --n_nodes 4 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.13013 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 0 --n_nodes 5 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.17351 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 0 --n_nodes 6 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.47716 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 0 --n_nodes 7 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 0 --n_nodes all --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 1 --n_nodes 3 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.4337 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 1 --n_nodes 4 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.13013 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 1 --n_nodes 5 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.17351 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 1 --n_nodes 6 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.47716 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 1 --n_nodes 7 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GIN --target-class 1 --n_nodes all --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 0 --n_nodes 4 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.13013 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 0 --n_nodes 6 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.47716 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 0 --n_nodes all --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 1 --n_nodes 3 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.4337 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 1 --n_nodes 4 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.13013 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 1 --n_nodes 5 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.17351 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 1 --n_nodes 6 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.47716 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 1 --n_nodes 7 --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset isAcyclic --GNN GCN --target-class 1 --n_nodes all --num_runs 1000  --lambda_1 0.2132 --lambda_2 0.73742 --lambda_3 0 --visual True --save True

# python3 pipeline.py --dataset MUTAG --GNN GIN --target-class 1 --num_runs 1000  --lambda_1 0.1504 --lambda_2 277.9444 --lambda_3 0.3724 --visual True --save True

# python3 pipeline.py --dataset MUTAG --GNN GCN --target-class 1 --num_runs 1000  --lambda_1 0.8356 --lambda_2 1582.7389 --lambda_3 0.02 --visual True --save True

# python3 pipeline.py --dataset highschool --target-class 0 --num_runs 1000  --lambda_1 1.08 --lambda_2 45352.8 --lambda_3 1511.76 --visual True --save True

# python3 pipeline.py --dataset highschool --target-class 1 --num_runs 1000  --lambda_1 0 --lambda_2 47242.5 --lambda_3 279.9556 --visual True --save True
