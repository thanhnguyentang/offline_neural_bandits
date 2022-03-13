# NeuraLCB 

This is the official JAX-based code for our `NeuraLCB` paper, ["Offline Neural Contextual Bandits: Pessimism, Optimization and Generalization"](https://openreview.net/pdf?id=sPIFuucA3F), 
ICLR 2022. `NeuraLCB` is a provably and computationally efficient offline policy learning (OPL) algorithm with deep neural networks: 
* Use a neural network to learn the reward
* Use neural network’s gradients for pessimistic exploitation
* Lower confidence bound strategy
* Stochastic gradient descent for optimization
* Stream offline data for generalization and adaptive offline data

## Dependencies 
* jax 
* optax 
* numpy 
* pandas 
* torchvision  

## Instruction 
* Run `NeuraLCB` and baseline methods in real-world datasets (MNIST and UCI Machine Learning Repository): 
  * non-parallelized version: `python realworld_main.py`
  * parallelized version: `python tune_realworld.py`
* Run `NeuraLCB` and baseline methods in synthetic datasets: 
  * non-parallelized version: `python synthetic_main.py`
  * parallelized version: `python tune_synthetic.py`

## Bibliography
```
@inproceedings{nguyen-tang2022offline,
title      =  {Offline Neural Contextual Bandits: Pessimism, Optimization and Generalization},
author     =  {Thanh Nguyen-Tang and 
               Sunil Gupta and 
               A. Tuan Nguyen and 
               Svetha Venkatesh},
booktitle  =  {International Conference on Learning Representations},
year       =  {2022},
url        =  {https://openreview.net/forum?id=sPIFuucA3F}
}
```
