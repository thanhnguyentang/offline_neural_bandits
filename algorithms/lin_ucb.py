"""Define Linear UCB offline bandit for data collection. """

import jax 
import jax.numpy as jnp
import optax
import numpy as np 
from easydict import EasyDict as edict
from jax.scipy.linalg import cho_factor, cho_solve
from core.bandit_algorithm import BanditAlgorithm 

import os 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
try:
    jnp.linalg.qr(jnp.array([[0, 1], [1, 1]]))
except RuntimeError:
    pass 

# When the context-action representation is encoded in the one-hot way, 
# LinLCB is equivalent to the disjoint linear model.  
class LinUCB(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='LinUCB'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        self.reset()

    def reset(self, seed=None):
        self.Sigma_hat = self.hparams.lambd0 * jnp.eye(self.hparams.context_dim * self.hparams.num_actions)
        self.y_hat = np.zeros((self.hparams.context_dim * self.hparams.num_actions,1)) 

    def sample_action(self, contexts):
        
        c, low = cho_factor(self.Sigma_hat)
        theta_hat = cho_solve((c, low), self.y_hat.reshape(-1,1))

        return batched_pi_linucb(contexts, theta_hat, self.Sigma_hat, self.hparams.beta)

    def update(self, contexts, actions, rewards): 
        for i in range(contexts.shape[0]):
            c = contexts[i,:].reshape(1,-1)
            a = actions.ravel()[i] 
            r = rewards.ravel()[i] 

            # @TODO: update Sigma_hat for all actions. 
            phi = jnp.kron(c, jax.nn.one_hot(a, self.hparams.num_actions).reshape(-1,1)).reshape(1,-1)

            self.Sigma_hat += phi.T @ phi 
            self.y_hat += r * phi.T 


"""Define util functions."""
@jax.jit 
def conf(A, b):
    """Compute (b.T A^-1 b)^(1/2)

    Args:
        A: (d,d) 
        b: (d,) 
        approx: bool. If True, A is (d,)
    Return: 
        scalar 
    """
    c, low = cho_factor(A)
    x = cho_solve((c, low), b.ravel())
    return jnp.sqrt(jnp.dot(b.ravel(), x.ravel()))  

@jax.jit 
def linucb_acq(context, action, theta_hat, Sigma_hat, beta):
    """Compute LCB in LinLCB

    Args:
        context: (d,) 
        action: scalar 
        theta_hat: (d,) 
        Sigma_hat: (d,d) 
        beta: float 
    Return:
        scalar 
    """
    num_actions = int(Sigma_hat.shape[0] / context.shape[0])
    phi = jnp.kron(context.reshape(1,-1), jax.nn.one_hot(action, num_actions).reshape(-1,1)).reshape(1,-1)
    
    f_hat = jnp.dot( phi.ravel(), theta_hat.ravel() )
    cnf = conf(Sigma_hat, phi)
    return f_hat + beta * cnf 

@jax.jit 
def pi_linucb(context, theta_hat, Sigma_hat, beta):
    """
    Args:
        context: (d,)
    
    Return:
        int in [0,K-1]
    """
    num_actions = int(Sigma_hat.shape[0] / context.shape[0]) 
    @jax.vmap 
    def linucb_action(a):
        return linucb_acq(context, a, theta_hat, Sigma_hat, beta) 
    return jnp.argmax(linucb_action(jnp.arange( num_actions )))

@jax.jit 
def batched_pi_linucb(contexts, theta_hat, Sigma_hat, beta):
    """
    Args:
        X: (n,d) 

    Return:
        (n,)
    """
    @jax.vmap 
    def linucb_actions(xs):
        return pi_linucb(xs, theta_hat, Sigma_hat, beta) 
    return linucb_actions(contexts) 



def test_LinUCB():
    hparams = edict({
        'context_dim': 8, 
        'num_actions': 2, 
        'beta': 1, 
        'lambd0': 0.1, 
    })

    linucb = LinUCB(hparams) 

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    num_contexts = 1

    context = jax.random.normal(key, (num_contexts,hparams.context_dim))
    action = jax.random.randint(subkey, (num_contexts,), minval=0, maxval = hparams.num_actions)
    key, subkey = jax.random.split(key)
    reward = jax.random.normal(key, (num_contexts,))

    linucb.update(context, action, reward)

    acts = linucb.sample_action(context)

    phi = jnp.kron(context, jax.nn.one_hot(action, hparams.num_actions).reshape(-1,1)).reshape(-1,hparams.context_dim * hparams.num_actions)
    
    print(acts)
    #print(phi)
    # print(linlcb.Sigma_hat)
    # print(linlcb.y_hat)



if __name__ == '__main__':
    test_LinUCB()