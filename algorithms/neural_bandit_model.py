"""Define  neural network models for (contextual) bandits.  

This model takes a context as input and predicts the rewards of all actions:

                            f(x,a) = < phi(x), w_a > 
"""

import numpy as np 
from core.nn import NeuralNetwork 
from core.utils import action_convolution 
import haiku as hk
import jax 
import jax.numpy as jnp
import optax
from tqdm import tqdm

class NeuralBanditModel(NeuralNetwork):
    """Build a neural network model for bandits.
    
    This model takes a context as input and predict the expected rewards of multiple actions. 
    """

    def __init__(self, optimizer, hparams, name):
        self.optimizer = optimizer 
        self.hparams = hparams 
        self.name = name 
        self.m = min(self.hparams.layer_sizes)
        self.build_model()
        print('{} has {} parameters.'.format(name, self.num_params))

    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""

        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        self.out = jax.jit(self.out_impure_fn) 
        self.grad_out = jax.jit(self.grad_out_impure_fn)
        self.loss = jax.jit(self.loss_impure_fn)
        self.update = jax.jit(self.update_impure_fn)

        # Initialize network parameters and opt states. 
        self.init()

        # Compute number of network parameters 
        p = (self.hparams.context_dim + 1) * self.hparams.layer_sizes[0] 
        for i in range(1, len(self.hparams.layer_sizes)):
            p += (self.hparams.layer_sizes[i-1] + 1) * self.hparams.layer_sizes[i]
        p += self.hparams.layer_sizes[-1] + 1 
        if self.hparams.layer_n: 
            p += sum(2 * self.hparams.layer_sizes)
        self.num_params = p 

    def net_impure_fn(self, context):
        """
        Args:
            context: context, (None, self.hparams.context_dim)
        """
        net_structure = []
        for num_units in self.hparams.layer_sizes:
            net_structure.append(
                hk.Linear(
                    num_units, w_init=hk.initializers.UniformScaling(self.hparams.s_init) 
                    )
                ) 
            if self.hparams.layer_n: 
                net_structure.append(hk.LayerNorm(axis=1, create_scale=True, create_offset=True))

            net_structure.append(self.hparams.activation) 

        net_structure.append(
                hk.Linear(self.hparams.num_actions, w_init=hk.initializers.UniformScaling(self.hparams.s_init) )
            )
        
        mlp = hk.Sequential(net_structure) 

        return mlp(context)
    
    def out_impure_fn(self, params, context):
        return self.nn.apply(params, context)

    def grad_out_impure_fn(self, params, context):
        """
        Args:
            params: Network parameters 
            context: (num_samples, context_dim)

        Return:
            action_grad_params: (num_actions, num_samples, p)
        """
        grad_params = jax.jacrev(self.out)(params, context)
        # return grad_params

        action_grad_params = []
        grad_param_leaves = jax.tree_leaves(grad_params)
        for a in range(self.hparams.num_actions):
            action_grad_param = []
        
            for grad_param in grad_param_leaves[:-2]:
                grad_param = grad_param.reshape((context.shape[0], self.hparams.num_actions, -1)) 
                action_grad_param.append( grad_param[:,a,:] ) 
            
            # for the last layer
            grad_param = grad_param_leaves[-2]
            action_grad_param.append( grad_param[:,a,a].reshape(-1,1)) 

            grad_param = grad_param_leaves[-1]
            action_grad_param.append( grad_param[:,a,:,a]) 

            action_grad_param = jnp.hstack(action_grad_param)
            action_grad_params.append(action_grad_param)

        return jnp.array(action_grad_params)


    def loss_impure_fn(self, params, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        preds = self.out(params, context) 

        squared_loss = 0.5 * jnp.mean(jnp.sum(action * jnp.square(preds - reward), axis=1), axis=0)
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params) 
            )

        return squared_loss + reg_loss 

    def update_impure_fn(self, params, opt_state, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        grads = jax.grad(self.loss)(params, context, action, reward)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def init(self): 
        key = jax.random.PRNGKey(self.hparams.seed)
        key, subkey = jax.random.split(key)

        context = jax.random.normal(key, (1, self.hparams.context_dim))
        self.params = self.nn.init(subkey, context) 
        self.opt_state = self.optimizer.init(self.params)

    def train(self, data, num_steps):
        if self.hparams.verbose:
            print('Training {} for {} steps.'.format(self.name, num_steps)) 
            
        params, opt_state = self.params, self.opt_state 
        for step in range(num_steps):
            x,w,y = data.get_batch_with_weights(self.hparams.batch_size) #(None,d), (None, num_actions), (None, num_actions)
            # print('DEBUG', x.shape, w.shape, y.shape)
            params, opt_state = self.update(params, opt_state, x,w,y) 

            if step % self.hparams.freq_summary == 0 and self.hparams.verbose:
                cost = self.loss(params, x,w,y)
                print('{} | step: {} | loss: {}'.format(self.name, step, cost))
        
        self.params, self.opt_state = params, opt_state


class NeuralBanditModelV2(NeuralBanditModel):
    """Build a neural network model V2 for bandits.
    
    This model takes an action-convoluted context as input and predict the expected reward of the context. 
    """

    def __init__(self, optimizer, hparams, name='NeuralBanditModelV2'):
        self.optimizer = optimizer 
        self.hparams = hparams 
        self.name = name 
        self.m = min(self.hparams.layer_sizes)
        self.build_model()
        print('{} has {} parameters.'.format(name, self.num_params))

    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""

        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        self.out = jax.jit(self.out_impure_fn) 
        self.grad_out = jax.jit(self.grad_out_impure_fn)
        self.action_convolution = jax.jit(self.action_convolution_impure_fn)
        self.loss = jax.jit(self.loss_impure_fn)
        self.update = jax.jit(self.update_impure_fn)

        # Initialize network parameters and opt states. 
        
        self.init(self.hparams.seed)

        # Compute number of network parameters 
        p = (self.hparams.context_dim + 1) * self.hparams.layer_sizes[0] 
        for i in range(1, len(self.hparams.layer_sizes)):
            p += (self.hparams.layer_sizes[i-1] + 1) * self.hparams.layer_sizes[i]
        p += self.hparams.layer_sizes[-1] + 1 
        if self.hparams.layer_n: 
            p += sum(2 * self.hparams.layer_sizes)
        self.num_params = p 

    def reset(self, seed):
        self.init(seed)

    def net_impure_fn(self, contexts, actions):
        """
        Args:
            convoluted_contexts: (None, self.hparams.context_dim * num_actions)
        """
        net_structure = []
        for num_units in self.hparams.layer_sizes:
            net_structure.append(
                hk.Linear(
                    num_units, w_init=hk.initializers.UniformScaling(self.hparams.s_init) 
                    )
                ) 
            if self.hparams.layer_n: 
                net_structure.append(hk.LayerNorm(axis=1, create_scale=True, create_offset=True))

            net_structure.append(self.hparams.activation) 

        net_structure.append(
                hk.Linear(1, w_init=hk.initializers.UniformScaling(self.hparams.s_init) )
            )
        
        mlp = hk.Sequential(net_structure) 

        convoluted_contexts = self.action_convolution(contexts, actions)
        return mlp(convoluted_contexts)

    def out_impure_fn(self, params, contexts, actions):
        return self.nn.apply(params, contexts, actions)

    def grad_out_impure_fn(self, params, contexts, actions):
        """
        Args:
            params: Network parameters 
            convoluted_contexts: (None, context_dim * num_actions)

        Return:
            grad_params: (None, p)
        """
        # grad_params = []
        # for gradp in jax.tree_leaves(jax.jacrev(self.out)(params, convoluted_contexts)):
        #     grad_params.append(gradp.reshape(convoluted_contexts.shape[0], -1))  
        # return jnp.hstack(grad_params)

        acts = jax.nn.one_hot(actions, self.hparams.num_actions)[:,None,:]
        ker = jnp.eye(self.hparams.context_dim)[None,:,:]
        sel = jnp.kron(acts, ker) # (None, context_dim, context_dim * num_actions, :)

        grad_params = jax.jacrev(self.out)(params, contexts, actions) 
        # return grad_params

        grads = []
        for key in grad_params:
            if key == 'linear': # extract the weights for the chosen actions only. 
                u = grad_params[key]['w'] 
                v = jnp.sum(jnp.multiply(u, sel[:,:,:,None]), axis=2) # (None, context_dim, :)
                grads.append(v.reshape(contexts.shape[0],-1))

                grads.append(grad_params[key]['b'].reshape(contexts.shape[0], -1)) 
            else:
                for p in jax.tree_leaves(grad_params[key]):
                    grads.append(p.reshape(contexts.shape[0], -1))  
                
        return jnp.hstack(grads)

    def action_convolution_impure_fn(self, contexts, actions):
        return action_convolution(contexts, actions, self.hparams.num_actions)

    def loss_impure_fn(self, params, contexts, actions, rewards):
        """
        Args:
            contexts: An array of context, (None, self.hparams.context_dim)
            actions: An array of actions, (None,)
            rewards: An array of rewards for the chosen actions, (None,)
        """
        preds = self.out(params, contexts, actions) 

        squared_loss = 0.5 * jnp.mean( jnp.square(preds.ravel() - rewards.ravel()) )
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_leaves(params) 
            )

        return squared_loss + reg_loss 

    def init(self, seed): 
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        context = jax.random.normal(key, (1, self.hparams.context_dim))
        action = jax.random.randint(subkey, shape=(1,), minval=0, maxval=self.hparams.num_actions) 

        self.params = self.nn.init(subkey, context, action) 
        self.opt_state = self.optimizer.init(self.params)

    
    def update_impure_fn(self, params, opt_state, contexts, actions, rewards):
        """
        Args:
            contexts: An array of contexts, (None, self.hparams.context_dim)
            actions: An array of actions, (None, )
            rewards: An array of rewards for the chosen actions, (None,)
        """
        grads = jax.grad(self.loss)(params, contexts, actions, rewards)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    
    def train(self, data, num_steps):
        if self.hparams.verbose:
            print('Training {} for {} steps.'.format(self.name, num_steps)) 
            
        params, opt_state = self.params, self.opt_state 
        for step in range(num_steps):
            x,a,y = data.get_batch(self.hparams.batch_size, self.hparams.data_rand) #(None,d), (None,), (None,)
            params, opt_state = self.update(params, opt_state, x,a,y) 

            if step % self.hparams.freq_summary == 0 and self.hparams.verbose:
                cost = self.loss(params, x,a,y)
                print('{} | step: {} | loss: {}'.format(self.name, step, cost))
        
        self.params, self.opt_state = params, opt_state
