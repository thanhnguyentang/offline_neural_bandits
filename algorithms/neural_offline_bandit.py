"""Define neural offline bandit algorithms. """

import math
import jax 
import jax.numpy as jnp
import optax
import numpy as np 
from joblib import Parallel, delayed
from core.bandit_algorithm import BanditAlgorithm 
from core.bandit_dataset import BanditDataset
from core.utils import inv_sherman_morrison, inv_sherman_morrison_single_sample, vectorize_tree
from algorithms.neural_bandit_model import NeuralBanditModel, NeuralBanditModelV2

class ExactNeuraLCBV2(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix and NeuralBanditModelV2. """
    def __init__(self, hparams, update_freq=1, name='ExactNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/hparams.lambd0 for _ in range(hparams.num_actions)
            ]
        ) # (num_actions, p, p)

    def reset(self, seed): 
        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/ self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
        ) # (num_actions, p, p)

        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
                # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                cnf = jnp.sqrt(gAg) # (num_samples,)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    
    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)
        
    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        # self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        # convoluted_contexts = self.nn.action_convolution(contexts, actions)  
        # u = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m)  # (num_samples, p)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            jax.ops.index_update(self.Lambda_inv, actions[i], \
                inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:]))

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
            
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))


class ApproxNeuraLCBV2(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix and NeuralBanditModelV2. """
    def __init__(self, hparams, update_freq=1, name='ApproxNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

        self.diag_Lambda = [
                jnp.ones(self.nn.num_params)* hparams.lambd0 for _ in range(hparams.num_actions)
            ]
         # (num_actions, p)

    def reset(self, seed): 
        self.diag_Lambda = [
                jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
         # (num_actions, p)

        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
                # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1) # (None, p) -> (None,)

                cnf = jnp.sqrt(gAg) # (num_samples,)
                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            # print(lcb)
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

        
        # def process(i):
        #     ctxs = contexts[i * cs: (i+1) * cs,:] 
        #     lcb = []
        #     for a in range(self.hparams.num_actions):
        #         actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

        #         f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
        #         # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
        #         g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
        #         gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1) # (None, p) -> (None,)

        #         cnf = jnp.sqrt(gAg) # (num_samples,)
        #         lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
        #         lcb.append(lcb_a.reshape(-1,1)) 
        #     lcb = jnp.hstack(lcb) 
        #     # print(lcb)
        #     return jnp.argmax(lcb, axis=1)
    
        # acts = Parallel(n_jobs=50,prefer="threads")(delayed(process)(i) for i in range(num_chunks))
        # return jnp.hstack(acts)

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        # Should run self.update_buffer before self.update to update the model in the latest data. 
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        # convoluted_contexts = self.nn.action_convolution(contexts, actions)  
        # u = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m)  # (num_samples, p)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            self.diag_Lambda[actions[i]] = self.diag_Lambda[actions[i]] + jnp.square(u[i,:])

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))


class NeuralGreedyV2(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='NeuralGreedyV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

    def reset(self, seed): 
        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        preds = []
        for a in range(self.hparams.num_actions):
            actions = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions) # (num_samples, 1)
            preds.append(f) 
        preds = jnp.hstack(preds) 
        return jnp.argmax(preds, axis=1)

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            context: An array of d-dimensional contexts
            action: An array of integers in [0, K-1] representing the chosen action 
            reward: An array of real numbers representing the reward for (context, action)
        
        """

        # self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        convoluted_contexts = self.nn.action_convolution(contexts, actions)

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        preds = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            preds.append(f) 
        preds = jnp.hstack(preds) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)

        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
            a, preds.ravel()[a], \
            cost, jnp.mean(jnp.square(norm)))
            )
        else:
            print('     r: {} | a: {} | f: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cost, jnp.mean(jnp.square(norm)))
                )
#===================================================================================================

class NeuraLCB(BanditAlgorithm):
    """NeuraLCB using diag approximation for confidence matrix. """
    def __init__(self, hparams, name='NeuraLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

        self.diag_Lambda = jnp.array(
                [hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions) ]
            ) # (num_actions, p)

    def sample_action(self, contexts):
        """
        Args:
            contexts: (None, self.hparams.context_dim)
        """
        n = contexts.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
            g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

            cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
            lcb = f - self.hparams.beta * cnf.T   # (num_samples, num_actions)
            return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = contexts[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
                lcb = f - self.hparams.beta * cnf.T   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            return jnp.array(acts)
            

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        g = self.nn.grad_out(self.nn.params, contexts)  # (num_actions, num_samples, p)
        g = jnp.square(g) / self.nn.m 
        for i in range(g.shape[1]): 
            self.diag_Lambda += g[:,i,:]

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

        cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | cnf: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), jnp.mean(jnp.square(params))))

class ExactNeuraLCB(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix. """
    def __init__(self, hparams, name='ExactNeuraLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

        # self.diag_Lambda = jnp.array(
        #         [hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions) ]
        #     ) # (num_actions, p)

        self.Lambda_inv = jnp.array(
            [
                np.eye(self.nn.num_params)/hparams.lambd0 for _ in hparams.num_actions
            ]
        ) # (num_actions, p, p)

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        n = contexts.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            f = self.nn.out(self.nn.params, context) # (num_samples, num_actions)
            g = self.nn.grad_out(self.nn.params, context) / jnp.sqrt(self.nn.m) # (num_actions, num_samples, p)
            gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
            cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

            lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
            return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = context[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
                cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

                lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            return jnp.array(acts)
            

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts, (None, context_dim)
            actions: An array of integers in [0, K-1] representing the chosen action, (None,)
            rewards: An array of real numbers representing the reward for (context, action), (None, )
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        u = self.nn.grad_out(self.nn.params, contexts) / jnp.sqrt(self.nn.m)  # (num_actions, num_samples, p)
        for i in range(g.shape[1]): 
            self.Lambda_inv =  inv_sherman_morrison(u[:,i,:], self.Lambda_inv)

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

        gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
        gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
        cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)
        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors for computing loss
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | cnf: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), jnp.mean(jnp.square(params))))


class NeuralGreedy(BanditAlgorithm):
    def __init__(self, hparams, name='NeuralGreedy'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

    def sample_action(self, contexts):
        f = self.nn.out(self.nn.params, contexts) # (num_contexts, num_actions)
        return jnp.argmax(f, axis=1)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            context: An array of d-dimensional contexts
            action: An array of integers in [0, K-1] representing the chosen action 
            reward: An array of real numbers representing the reward for (context, action)
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))
        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)

        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)
        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), jnp.mean(jnp.square(params))))
