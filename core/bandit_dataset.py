"""Define a data buffer for contextual bandit algorithms. """

import numpy as np 
import jax
import jax.numpy as jnp 

class BanditDataset(object):
    """Append new data and sample random minibatches. """
    def __init__(self, context_dim, num_actions, buffer_s = -1, name='bandit_data'):
        """
        Args:
            buffer_s: Buffer size. Only last buffer_s will be used. 
                If buffer_s = -1, all data is used. 
        """
        self.name = name 
        self.context_dim = context_dim 
        self.num_actions = num_actions 
        self.buffer_s = buffer_s 
        self.contexts = None # An array of d-dim contexts
        self.actions = None # An array of actions 
        self.rewards = None # An array of one-hot rewards 

    def reset(self):
        self.contexts = None 
        self.actions = None 
        self.rewards = None 

    def add(self, context, action, reward):
        """Add one or multiple samples to the buffer. 

        Args:
            context: An array of d-dimensional contexts, (None, context_dim)
            action: An array of integers in [0, K-1] representing the chosen action, (None,1)
            reward: An array of real numbers representing the reward for (context, action), (None,1)

        """
        c = context.reshape(-1, self.context_dim) 
        if self.contexts is None:
            self.contexts = c 
        else:
            self.contexts = jnp.vstack((self.contexts, c))

        r = jax.nn.one_hot(action.ravel(), self.num_actions) * reward.reshape(-1,1)

        if self.rewards is None: 
            self.rewards = r 
        else: 
            self.rewards = jnp.vstack((self.rewards, r)) 

        if self.actions is None: 
            self.actions = action.reshape(-1,1) 
        else:
            self.actions = jnp.vstack((self.actions, action.reshape(-1,1)))

    def get_batch_with_weights(self, batch_size):
        """
        Return:
            x: (batch_size, context_dim)
            w: (batch_size, num_actions)
            y: (batch_size, num_actions)
        """
        n = self.num_samples 
        if self.buffer_s == -1:
            ind = np.random.choice(range(n), batch_size) 
        else: 
            ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        return self.contexts[ind, :], jax.nn.one_hot(self.actions[ind,:].ravel(), self.num_actions), self.rewards[ind, :]

    def get_batch(self, batch_size, rand=True):
        """
        Return:
            x: (batch_size, context_dim)
            a: (batch_size, )
            y: (batch_size, )
        """
        n = self.num_samples 
        assert n > 0 
        if rand:
            if self.buffer_s == -1:
                ind = np.random.choice(n, batch_size) 
            else: 
                ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        else:
            ind = range(n - batch_size,n)
        a = self.actions[ind,:].ravel()
        return self.contexts[ind, :], a, self.rewards[ind, a]

    @property 
    def num_samples(self): 
        return 0 if self.contexts is None else self.contexts.shape[0]


def test_BanditDataset():
    context_dim = 5 
    num_actions = 2 
    n = 8 
    bd = BanditDataset(context_dim, num_actions)


    contexts = np.random.uniform(size=(n, context_dim)) 
    actions = np.random.randint(0,num_actions, (n,1)) 
    rewards = np.random.randn(n,1) 

    print(contexts.shape, actions.shape, rewards.shape)

    bd.add(contexts, actions, rewards) 

    print(bd.contexts.shape, bd.actions.shape, bd.rewards.shape)

    print(bd.actions)
    print(bd.rewards)
    print('=======')
    c,w,r = bd.get_batch_with_weights(batch_size=1)
    print(c.shape, w.shape, r.shape)
    print(w)
    print(r)


# test_BanditDataset()