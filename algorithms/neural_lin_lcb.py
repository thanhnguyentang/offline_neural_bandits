"""Define Neural Linear LCB offline bandit. """

import jax 
import jax.numpy as jnp
import optax
import numpy as np 
import math 
from jax.scipy.linalg import cho_factor, cho_solve
from core.bandit_algorithm import BanditAlgorithm 
from core.utils import inv_sherman_morrison, inv_sherman_morrison_single_sample, vectorize_tree
from algorithms.neural_bandit_model import NeuralBanditModel, NeuralBanditModelV2


class ExactNeuralLinLCBV2(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='ExactNeuralLinLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(0.0001) # dummy
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        
        self.reset(self.hparams.seed)

    def reset(self, seed): 
        self.y_hat = jnp.zeros(shape=(self.hparams.num_actions, self.nn.num_params))

        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/ self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
        ) # (num_actions, p, p)

        self.nn.reset(seed) 

    def sample_action(self, contexts):

        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                cnf = jnp.sqrt(gAg) # (num_samples,)

                f = jnp.dot(gA, self.y_hat[a,:]) 

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    def update(self, contexts, actions, rewards): 
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            jax.ops.index_update(self.y_hat, actions[i], \
                self.y_hat[actions[i]] +  rewards[i] * u[i,:])
            jax.ops.index_update(self.Lambda_inv, actions[i], \
                inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:]))

class ApproxNeuralLinLCBV2(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='ApproxNeuralLinLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(0.0001) # dummy
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        
        self.reset(self.hparams.seed)

    def reset(self, seed): 
        # self.y_hat = jnp.zeros(shape=(self.hparams.num_actions, self.nn.num_params))

        self.y_hat = [
            jnp.zeros(shape=(self.nn.num_params,)) for _ in range(self.hparams.num_actions)
        ]

        self.diag_Lambda = [
                jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
         # (num_actions, p)

        self.nn.reset(seed) 

    def sample_action(self, contexts):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)

                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)                
                cnf = jnp.sqrt(gAg) # (num_samples,)

                f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    # def update_buffer(self, contexts, actions, rewards): 
    #     self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards): 
        # print(rewards)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            v = self.diag_Lambda[actions[i]] 
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            self.diag_Lambda[actions[i]] = jnp.square(u[i,:]) + self.diag_Lambda[actions[i]]

            self.y_hat[actions[i]] = self.y_hat[actions[i]] +  rewards[i] * u[i,:] 
            # jax.ops.index_add(self.y_hat, actions[i], rewards[i] * u[i,:])

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(self.nn.params)))

        preds = []
        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gAg = jnp.sum(jnp.square(g) / self.diag_Lambda[a][:], axis=-1)                
            cnf = jnp.sqrt(gAg) # (num_samples,)

            f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)

            cnfs.append(cnf) 
            preds.append(f)
        cnf = jnp.hstack(cnfs) 
        preds = jnp.hstack(preds)

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

class ApproxNeuralLinGreedyV2(ApproxNeuralLinLCBV2):
    def __init__(self, hparams, update_freq=1,name='ApproxNeuralLinGreedyV2'):
        super().__init__(hparams, update_freq,name)
 
    def sample_action(self, contexts):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)

                f = jnp.sum(jnp.multiply(g, self.y_hat[a][:]) / self.diag_Lambda[a][:], axis=-1)

                lcb_a = f.ravel() # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            # print(self.y_hat.sum())
            # print(lcb)
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)


class ApproxNeuralLinLCBJointModel(BanditAlgorithm):
    """Use joint model as LinLCB. 
    
    Sigma_t = lambda I + \sum_{i=1}^t phi(x_i,a_i) ph(x_i,a_i)^T  
    theta_t = Sigma_t^{-1} \sum_{i=1}^t phi(x_i,a_i) y_i ""
    """
    def __init__(self, hparams, update_freq=1, name='ApproxNeuralLinLCBJointModel'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(0.0001) # dummy
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        
        self.reset(self.hparams.seed)

    def reset(self, seed): 
        # self.y_hat = jnp.zeros(shape=(self.hparams.num_actions, self.nn.num_params))
        self.y_hat = jnp.zeros(shape=(self.nn.num_params,)) 
        self.diag_Lambda = jnp.ones(self.nn.num_params) * self.hparams.lambd0  
        self.nn.reset(seed) 

    def sample_action(self, contexts):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)

                gAg = jnp.sum(jnp.square(g) / self.diag_Lambda, axis=-1)                
                cnf = jnp.sqrt(gAg) # (num_samples,)

                f = jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    # def update_buffer(self, contexts, actions, rewards): 
    #     self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards): 
        # print(rewards)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            self.diag_Lambda = jnp.square(u[i,:]) + self.diag_Lambda
            self.y_hat = self.y_hat +  rewards[i] * u[i,:] 
        

class NeuralLinGreedyJointModel(ApproxNeuralLinLCBJointModel):
    def __init__(self, hparams, update_freq=1, name='NeuralLinGreedyJointModel'):
        super().__init__(hparams, update_freq, name)
    def sample_action(self, contexts):
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)
                f = jnp.sum(jnp.multiply(g, self.y_hat) / self.diag_Lambda, axis=-1)
                lcb_a = f.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

#=======================        
class ExactNeuralLinGreedyV2(ExactNeuralLinLCBV2):
    def __init__(self, hparams, update_freq=1, name='ExactNeuralLinGreedyV2'):
        super().__init__(hparams, update_freq, name)

    def sample_action(self, contexts):

        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            preds = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m) # (None, p)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                # gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                # cnf = jnp.sqrt(gAg) # (num_samples,)

                f = jnp.dot(gA, self.y_hat[a,:]) 

                preds.append(f.reshape(-1,1)) 
            preds = jnp.hstack(preds) 
            acts.append( jnp.argmax(preds, axis=1)) 
        return jnp.hstack(acts)

#=============================================================================================
class NeuralLinLCB(BanditAlgorithm):
    def __init__(self, hparams, name='NeuralLinLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(0.0001) # dummy
        self.nn = NeuralBanditModel(opt, hparams, 'init_nn')

        self.Sigma_hat = hparams.lambd * jnp.ones(self.nn.num_params)
        # else:
        #     self.Sigma_hat = hparams.lambd * jnp.eye(self.nn.p)
        self.y_hat = 0 

    def action(self, contexts):
        n = contexts.shape[0] 
        if n <= self.hparams.max_test_batch:
            phi = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)
            f_hat = jnp.sum(phi * self.y_hat.reshape(1,1,-1) / self.Sigma_hat.reshape(1,1,-1), axis=-1) # (num_actions, num_samples)
            cnf = jnp.square(jnp.sum(jnp.square(phi) / self.Sigma_hat.reshape(1,1,-1), axis=-1)) # (num_actions, num_samples)
            lcb = f_hat - self.hparams.beta * cnf 
            return jnp.argmax(lcb, axis=0)
        else: # Break contexts in batches if it is large. 
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = contexts[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                phi = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)
                f_hat = jnp.sum(phi * self.y_hat.reshape(1,1,-1) / self.Sigma_hat.reshape(1,1,-1), axis=-1) # (num_actions, num_samples)
                cnf = jnp.square(jnp.sum(jnp.square(phi) / self.Sigma_hat.reshape(1,1,-1), axis=-1)) # (num_actions, num_samples)
                lcb = f_hat - self.hparams.beta * cnf 
                acts.append(jnp.argmax(lcb, axis=0).ravel())
            return jnp.array(acts)

    def update(self, context, action, reward): 
        for i in range(context.shape[0]):
            c = context[i,:].reshape(1,-1)
            a = action.ravel()[i] 
            r = reward.ravel()[i] 

            g_out = self.nn.grad_out(self.nn.params, c)

            phi = g_out[a,:,:].reshape(1,-1)

            self.Sigma_hat = self.Sigma_hat.ravel() + jnp.square(phi).ravel() / jnp.sqrt(self.nn.m)
            # else:
            #     self.Sigma_hat += phi.T @ phi / jnp.sqrt(self.nn.m)
            self.y_hat += r * phi.T 


class NeuralLinGreedy(NeuralLinLCB):
    def action(self, contexts):
        n = contexts.shape[0] 
        if n <= self.hparams.max_test_batch:
            phi = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)
            f_hat = jnp.sum(phi * self.y_hat.reshape(1,1,-1) / self.Sigma_hat.reshape(1,1,-1), axis=-1) # (num_actions, num_samples)
            return jnp.argmax(f_hat, axis=0)
        else: # Break contexts in batches if it is large. 
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = contexts[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                phi = self.nn.grad_out(self.nn.params, c)
                f_hat = jnp.sum(phi * self.y_hat.reshape(1,1,-1) / self.Sigma_hat.reshape(1,1,-1), axis=-1) 
                acts.append(jnp.argmax(f_hat, axis=0).ravel())
            return jnp.array(acts)
