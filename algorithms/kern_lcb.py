"""Define Kernel LCB offline bandit. """

import jax 
import jax.numpy as jnp
import numpy as np 
from easydict import EasyDict as edict
from jax.scipy.linalg import cho_factor, cho_solve
from core.bandit_algorithm import BanditAlgorithm 
# from sklearn_jax_kernels import ConstantKernel, RBF, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

class KernLCB(BanditAlgorithm):
    def __init__(self, hparams, update_freq = 1, name='KernLCB'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 

        self.reset()

    def reset(self, seed=None):
        # kern = ConstantKernel(1.0) * RBF(1.0)
        kern = RBF(self.hparams.rbf_sigma, length_scale_bounds ='fixed')
        self.gpr = GaussianProcessRegressor(kernel = kern, alpha=1e-6)
        self.X = None 
        self.Y = None

    def sample_action(self, contexts):
        """
        Args:
            X_test: (n,d)
            gpr:
            dm: DataManifold
            kern_beta:
        """
        c = jnp.kron(contexts.reshape(-1,1,1, self.hparams.context_dim), \
                jnp.eye(self.hparams.num_actions).reshape(1, self.hparams.num_actions, self.hparams.num_actions,1 ) )

        c = c.reshape(contexts.shape[0], self.hparams.num_actions, -1) # (n,K,dK)
        c_flat = c.reshape(-1, self.hparams.context_dim * self.hparams.num_actions)

        mus, sigmas = self.gpr.predict(c_flat, return_std=True) #(nK,) 
        res = mus.ravel() - self.hparams.beta * sigmas.ravel() #(nK)
        res = res.reshape(-1, self.hparams.num_actions)
        return jnp.argmax(res, axis=1)

    def update_buffer(self, context, action, reward): 
        # Update only when the number of samples <= 1000
        if self.X is None or self.X.shape[0] <= 1000:
            context = context.reshape(-1, self.hparams.context_dim) 
            action = action.ravel() 
            reward = reward.ravel()

            c = jnp.kron(context.reshape(-1,1,1, self.hparams.context_dim), \
                    jnp.eye(self.hparams.num_actions).reshape(1, self.hparams.num_actions, self.hparams.num_actions,1 ) )

            c = c.reshape(context.shape[0], self.hparams.num_actions, -1) # (n,K,dK)
            c = c[jnp.arange(context.shape[0]), action, :] # (n,dK)

            if self.X is None: 
                self.X = c 
                self.Y = reward 
            else: 
                self.X = jnp.vstack((self.X, c)) 
                self.Y = jnp.vstack((self.Y, reward))

            # 
    def update(self, contexts=None, actions=None, rewards=None):
        self.gpr.fit(self.X, self.Y) 
            


def test_KernLCB():
    hparams = edict({
        'context_dim': 8, 
        'num_actions': 2, 
        'beta': 1, 
    })

    kernlcb = KernLCB(hparams) 

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    context = jax.random.normal(key, (5,hparams.context_dim))
    action = jax.random.randint(subkey, (5,), minval=0, maxval = hparams.num_actions)
    key, subkey = jax.random.split(key)
    reward = jax.random.normal(key, (5,))

    kernlcb.update(context, action, reward)

    acts = kernlcb.action(context)

    print(acts)
