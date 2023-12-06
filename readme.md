# Towards Meta-Pruning via Optimal Transport

## Introduction
This repository provides the code behind Intra-Fusion.

## Requirements
- Python (v3.10.8)
- torch (v1.13.0)
- torchvision (v0.14.0)
- sklearn (v1.1.3)
- pot (v0.8.2)
- numpy (v1.24.2)
- pandas (v1.5.3)

## Explanation
As explained in the appendix of the paper, we build Intra-Fusion as an extension of an existing pruning library: https://github.com/VainF/Torch-Pruning. In order to not get lost in codebase, we hereby point to the parts that relate to Intra-Fusion.

### Merging Batch-Normalization with the prior layer
File: torch_pruning/pruner/algorithms/metapruner.py
Function: merge_bn()
Here, the batchnormalization layer is merged with the layer it acts upon, thereby preserving the function output

### Derivation of the Optimal Transport Map
File: torch_pruning/optimal_transport.py
Class: OptimalTransport()
Calling this class will provide the optimal transport map that is later used to fuse maching neurons.

### Layer compression
File: torch_pruning/function.py
Function (line 96): _prune_parameter_and_grad()
Here, we are given the transport map, and followingly derive the compressed layer via matrix multiplication with the optimal transport map (ot_map).

## Run
In order to reproduce our results for a few examples, please run the test.py file and change configurations to your liking.


