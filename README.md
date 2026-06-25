# Evolutionary QML Circuit Architecture Search

This script implements a genetic algorithm to optimize the partition between classical and quantum layers in a hybrid neural network, based on the PennyLane PyTorch tutorial (https://pennylane.ai/demos/tutorial_qnn_module_torch).

## How it works
Each genome is a binary sequence where 0 = classical layer and 1 = quantum layer.
The genetic algorithm evaluates different architectures, selects the best ones, and combines them to find an optimal partition.

## Results
### Run 1
```
[0, 0, 0, 1, 1, 0] → 0.500 ⚠ UNTRAINED
[0, 0, 0, 0, 0, 0] → 0.840 ✗ WORST
[1, 1, 1, 0, 0, 0] → 0.870 
[0, 1, 1, 0, 0, 0] → 0.885 
[1, 1, 1, 0, 1, 0] → 0.945 ★ BEST
```
### Run 2
```
[1, 0, 0, 0, 1, 1] → 0.500 ⚠ UNTRAINED
[0, 0, 0, 0, 0, 0] → 0.840 ✗ WORST
[1, 0, 0, 0, 0, 0] → 0.850 
[1, 0, 1, 1, 1, 0] → 0.955 ★ BEST
[1, 0, 1, 0, 0, 0] → 0.955 ★ BEST
```
