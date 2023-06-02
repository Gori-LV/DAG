# DAG-Explainer

This is the official implement of the paper _On Data-Aware Global Explainability of Graph Neural Networks_.

[//]: # (![our_work]&#40;/intro_eg.png&#41;)
<p align="center">
  <img src="https://github.com/Gori-LV/DAG/blob/main/workflow.png" />

[//]: # (    Figure. Workflow of DAG-Explainer.)
</p>

[//]: # ([On Explainability of Graph Neural Networks via Subgraph Explorations]&#40;https://arxiv.org/abs/2102.05152&#41;)


## Installation
* Clone the repository 
* Create the env and install the requirements

```shell script
$ git clone https://github.com/Gori-LV/DAG
$ cd DAG
$ source ./install.sh
```

## Usage
* Download the required [datasets](https://hkustconnect-my.sharepoint.com/:f:/g/personal/glvab_connect_ust_hk/EqFR8NjD49tLtPp9TgicvjQBxkj_15wDT4D2fdrJ6Adx2A?e=P9NeHI) to `/data`
* Download the [checkpoints](https://hkustconnect-my.sharepoint.com/:f:/g/personal/glvab_connect_ust_hk/EscGZSmy_W9KpSWE-cxk6yQB2_g3RYvO-LypseIN-X8Ngg) to `/checkpoints`
* Run the searching scripts with corresponding dataset.
```shell script
$ source ./scripts.sh
``` 
The hyper-parameters used for different datasets are shown in this script.


## Examples
We provide examples on how to use DAG-Explainer on the three dataset. Run `*.ipynb` files in Jupyter Notebook or Jupyter Lab. 