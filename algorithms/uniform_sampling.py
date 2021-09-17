"""Define a uniform sampling algorithm. """

import jax.numpy as jnp
import optax
import numpy as np 
from core.bandit_algorithm import BanditAlgorithm 
from core.bandit_dataset import BanditDataset

class UniformSampling(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='Uniform'):
        self.name = name 
        self.hparams = hparams
        self.update_freq = update_freq 

    def sample_action(self, contexts):
        return np.random.randint(0, self.hparams.num_actions, (contexts.shape[0],)) 