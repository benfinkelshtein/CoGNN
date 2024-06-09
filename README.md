# Cooperative Graph Neural Networks #

This repository contains the official code base of the paper **[Cooperative Graph Neural Networks](https://arxiv.org/abs/2310.01267)**, accepted to ICML 2024

## Installation ##
To reproduce the results please use Python 3.9, PyTorch version 2.0.0, Cuda 11.8, PyG version 2.3.0, and torchmetrics.

```bash
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.3.0
pip install torchmetrics ogb rdkit
pip install matplotlib
```

## Datasets

### Synthetic datasets

The synthetic datasets RootNeighbours and Cycles should be generated with a seed of 0.

### Node Classification

Available in Pytorch Geometric.

### Graph Classification

The TUDatasets and Social network graph classification datasets can be found at ``datasets/``.

Make sure to unzip the REDDIT-MULTI-5K.zip first!
```
unzip datasets/REDDIT-MULTI-5K.zip -d datasets
```

### LRGB

The Peptides-func dataset can be found at ``datasets/peptides-functional``.

## Running

The script we use to run the experiments is ``./main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

The parameters of the script are:

- ``--dataset``: name of the dataset.
- ``--pool``: name of the graph pooling.
- ``--learn_temp``: a flag to be used when the temperature is learned. 
- ``--temp_model_type``: the type of GNN that learns a temperature for the Gumbel-softmax estimator (relevant only if ``learn_temp`` is present). 
- ``--tau0``: the tau0 parameter in a learnable temperature model (relevant only if ``learn_temp`` is present). 
- ``--temp``: the temperature of the Gumbel-softmax estimator (relevant only if ``learn_temp`` is not present). 
- ``--max_epochs``: the number of epochs. 
- ``--batch_size``: the batch size. 
- ``--lr``: the learn rate. 
- ``--env_model_type``: the type of GNN the environment network uses.
- ``--env_num_layers``: the environment network's number of layers.
- ``--env_dim``: the environment network's hidden dimension.
- ``--skip``: a flag that is used to include skip connections.
- ``--batch_norm``: the batch size.
- ``--layer_norm``: a flag that is used to include layer_norm.
- ``--dec_num_layers``: the number of layers the decoder MLP uses.
- ``--pos_enc``: the type of positional encoding used.
- ``--act_model_type``: the type of GNN the action network uses.
- ``--act_num_layers``: the action network's number of layers.
- ``--act_dim``: the action network's hidden dimension.
- ``--seed``: a seed to set random processes.
- ``--gpu``: the number of the gpu that is used to run the code on.
- ``--fold``: a specific fold of the dataset (only applicable to a portion of the datasets used).
- ``--weight_decay``: the weight decay.
- ``--step_size``: the step_size of the ``StepLR`` scheduler (only applicable to a portion of the datasets used).
- ``--gamma``: the gamma of the ``StepLR`` scheduler (only applicable to a portion of the datasets used).
- ``--num_warmup_epochs``: the num_warmup_epochs of ``cosine_with_warmup_scheduler`` (only applicable to a portion of the datasets used).

## Example running

To perform experiments with a CoGNN($`\mu, \mu`$) model with 3 environment layers and an environment hidden dimension of 64, with a 1-layer action network with a hidden dimension of 16.  See an example for the use of the following command: 
```bash
python -u main.py --dataset roman_empire --env_model_type MEAN_GNN --act_model_type MEAN_GNN --env_dim 64 --env_num_layers 3 --act_dim 16 --act_num_layers 1 --seed 0
```
## Cite

If you make use of this code, or its accompanying [paper](https://arxiv.org/abs/2310.01267), please cite this work as follows:
```bibtex
@inproceedings{finkelshtein2023cooperative,
  title = "Cooperative Graph Neural Networks",
  author = "Ben Finkelshtein and Xingyue Huang and Michael Bronstein and {\.I}smail {\.I}lkan Ceylan",
  year = "2024",
  booktitle = "Proceedings of Forty-first International Conference on Machine Learning (ICML)",
  url = "https://arxiv.org/abs/2310.01267",
}
```
