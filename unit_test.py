import jax 
import jax.numpy as jnp 
import optax 
from easydict import EasyDict as edict
import numpy as np
import time  

from algorithms.lin_lcb import test_LinLCB
from algorithms.kern_lcb import test_KernLCB
from algorithms.neural_lin_lcb import NeuralLinLCB
from algorithms.neural_bandit_model import NeuralBanditModelV2 
from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2
from core.bandit_dataset import BanditDataset

from core.utils import action_convolution, inv_sherman_morrison, inv_sherman_morrison_single_sample, sample_offline_policy


def test_NeuralLinLCB():
    hparams = edict({
        'layer_sizes': [100,100], 
        's_init': 1, 
        'activation': jax.nn.relu, 
        'layer_n': True,
        'context_dim': 32, 
        'num_actions': 2, 
        'beta': 1, 
        'lambd': 0.1, 
        'seed': 0,
    })

    neural_linlcb = NeuralLinLCB(hparams) 

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    context = jax.random.normal(key, (5,hparams.context_dim))
    action = jax.random.randint(subkey, (5,), minval=0, maxval = hparams.num_actions)
    key, subkey = jax.random.split(key)
    reward = jax.random.normal(key, (5,))

    neural_linlcb.update(context, action, reward)

    acts = neural_linlcb.action(context)

    print(acts)

def test_action_convolution():
    context_dim = 3 
    num_actions = 2 
    num_contexts = 4

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    contexts = jax.random.normal(key, (num_contexts,context_dim))
    actions = jax.random.randint(subkey, (num_contexts,), minval=0, maxval = num_actions)

    convoluted_contexts = action_convolution(contexts, actions, num_actions)

    print(convoluted_contexts.shape)
    print(convoluted_contexts)
    print(actions)
    print(contexts)

def test_sample_offline_policy():
    context_dim = 3 
    num_actions = 2 
    num_contexts = 4

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    contexts = jax.random.normal(key, (num_contexts,context_dim))
    rewards = jax.random.normal(subkey, (num_contexts,num_actions))

    actions = sample_offline_policy(mean_rewards=rewards, num_contexts=num_contexts, num_actions=num_actions, pi='online', 
                contexts=contexts, rewards=rewards)

    print(actions)

def test_NeuralBanditModelV2():
    hparams = edict({
        'layer_sizes': [10,10], 
        's_init': 1, 
        'activation': jax.nn.relu, 
        'layer_n': True,
        'context_dim': 3, 
        'num_actions': 2, 
        'beta': 1, 
        'lambd': 0.1, 
        'seed': 0,
        'lr': 0.001, 
        'verbose': True, 
        'batch_size': 32,
        'freq_summary': 10, 
        'chunk_size': 100, 
        'lambd0': 1, 
        'num_steps': 100, 
        'buffer_s': -1,
        'debug_mode': 'simple'
    })

    # optimizer = optax.adam(hparams.lr)

    # nn = NeuralBanditModelV2(optimizer, hparams, name='NeuralBanditModelV2')

    ## Data 
    context_dim = hparams.context_dim
    num_actions = hparams.num_actions 
    num_contexts = 5

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    contexts = jax.random.normal(key, (num_contexts,context_dim))
    actions = jax.random.randint(subkey, (num_contexts,), minval=0, maxval = num_actions)
    key, subkey = jax.random.split(key)
    rewards = jax.random.normal(key, (num_contexts,)) 
    convoluted_contexts = action_convolution(contexts, actions, num_actions)

    ## Test grad_out
    # gout = nn.grad_out(nn.params, convoluted_contexts)

    # for g in jax.tree_leaves(gout): 
    #     print(g.shape)   

    # print(gout)

    # u = gout['linear']['w'] 

    # print(u.shape)

    # acts = jax.nn.one_hot(actions, hparams.num_actions)[:,None,:]
    # ker = jnp.eye(hparams.context_dim)[None,:,:]

    # sel = jnp.kron(acts, ker) # (None, context_dim, context_dim * num_actions, :)
    # print(sel.shape)
    # print(sel)

    # v = jnp.sum(jnp.multiply(u, sel[:,:,:,None]), axis=2) # (None, context_dim, :)
    # print(v.shape) 
    # print(v)
    # print('------')
    # print(u)
   
    # print('=====')
    # params = []
    # for g in gout:
    #     if g == 'linear': 
    #         u = gout[g]['w'] 
    #         v = jnp.sum(jnp.multiply(u, sel[:,:,:,None]), axis=2) # (None, context_dim, :)

    #         params.append(v.reshape(num_contexts,-1)) 
    #         params.append(gout[g]['b'].reshape(num_contexts, -1)) 
    #     else:
    #         for p in jax.tree_leaves(gout[g]):
    #             params.append(p.reshape(num_contexts, -1))  
            
    # params = jnp.hstack(params)
    # print(params.shape) 
    # print(params[0])

    ## Test loss 
    # cost = nn.loss(nn.params, contexts, actions, rewards) 
    # print(cost)

    ## Test train 
    # data = BanditDataset(context_dim=hparams.context_dim, num_actions=hparams.num_actions, buffer_s=-1)
    # data.add(contexts, actions, rewards) 

    # nn.train(data, num_steps=100)

    # x,a,y = data.get_batch(2) 
    # print(x.shape) 
    # print(a.shape) 
    # print(y.shape) 
    # print(x) 
    # print(a) 
    # print(y)

    # print(contexts) 
    # print(actions) 
    # print(rewards) 

    ## Test ExactNeuraLCBV2 
    # exactneuralcbV2 = ExactNeuraLCBV2(hparams)

    # print(exactneuralcbV2.Lambda_inv.shape)
    # exactneuralcbV2.update(contexts, actions, rewards) 

    # pred_actions = exactneuralcbV2.sample_action(contexts) 
    # print(pred_actions.shape)
    # print(pred_actions)

    # exactneuralcbV2.monitor(contexts[0:1,:], actions[0:1], rewards[0:1])

    ## Test NeuralGreedyV2 
    # neuralgreedyV2 = NeuralGreedyV2(hparams) 
    # neuralgreedyV2.update(contexts, actions, rewards) 

    # pred_actions = neuralgreedyV2.sample_action(contexts) 
    # print(pred_actions.shape)
    # print(pred_actions)

    # neuralgreedyV2.monitor(contexts[0:1,:], actions[0:1], rewards[0:1])

    ## Test ApproxNeuraLCBV2
    algo = ApproxNeuraLCBV2(hparams)
    algo.update_buffer(contexts,actions,rewards)
    algo.update(contexts, actions, rewards)
    print(algo.diag_Lambda.shape)

    pred_actions = algo.sample_action(contexts) 
    print(pred_actions)


def test_inv_sherman_morrison():
    d = 5 
    n = 1
    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    # u = jax.random.normal(key, (n,d)) 
    # A_inv = jnp.array([
    #     jnp.eye(d) for _ in range(n)
    # ])
    u = jax.random.normal(key, (d,)) 
    A_inv = jnp.eye(d)

    B = inv_sherman_morrison_single_sample(u, A_inv) 
    print(B.shape)
    print(B)


    # C = jnp.linalg.inv(A_inv[0,:,:] + u.T @ u)
    # print(C)
    
    # D = u.T @ u 
    # print(u) 
    # print(D) 
    # print(D.T)
     
def test_ApproxNeuraLCBV2():
    context_dim = 784 
    num_actions = 10 
    hparams = edict({
        'layer_sizes': [100,100], 
        's_init': 1, 
        'activation': jax.nn.relu, 
        'layer_n': True,
        'seed': 0,
        'context_dim': context_dim, 
        'num_actions': num_actions, 
        'beta': 0.1, # [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        'lambd': 1e-4, # regularization param: [0.1m, m, 10 m  ]
        'lr': 1e-4, 
        'lambd0': 0.1, # shoud be lambd/m in theory but we fix this at 0.1 for simplicity and mainly focus on tuning beta 
        'verbose': False, 
        'batch_size': 50,
        'freq_summary': 100, 
        'chunk_size': 1, 
        'num_steps': 100,
        'test_freq': 100,  
        'buffer_s': -1, 
        'data_rand': True,
        'debug_mode': 'full' # simple/full
    })
    ## Data 
    context_dim = hparams.context_dim
    num_actions = hparams.num_actions 
    num_contexts = 5
    num_test_contexts = 1000

    key = jax.random.PRNGKey(0) 
    key, subkey = jax.random.split(key)

    contexts = jax.random.normal(key, (num_contexts,context_dim))
    actions = jax.random.randint(subkey, (num_contexts,), minval=0, maxval = num_actions)
    key, subkey = jax.random.split(key)
    rewards = jax.random.normal(key, (num_contexts,)) 

    algo = ApproxNeuraLCBV2(hparams)
    algo.update_buffer(contexts,actions,rewards)
    # algo.update(contexts, actions, rewards)
    print(algo.diag_Lambda)

    test_contexts = jax.random.normal(key, (num_test_contexts,context_dim))
    test_actions = jax.random.randint(subkey, (num_test_contexts,1), minval=0, maxval = num_actions)
    # t1 = time.time()
    # conv_actions = action_convolution(test_contexts, test_actions,num_actions)
    # print(time.time() - t1)
    # actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

    # for _ in range(5):
    #     t1 = time.time()
    #     f = algo.nn.out(algo.nn.params, test_contexts, test_actions) # (num_samples, 1)
    #     t2 = time.time()
    #     # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
    #     g = algo.nn.grad_out(algo.nn.params, test_contexts, test_actions) / jnp.sqrt(algo.nn.m)
    #     t3 = time.time()
    #     gAg = jnp.sum( jnp.square(g) / algo.diag_Lambda[0][:], axis=-1) 
    #     t4 = time.time()
    #     cnf = jnp.sqrt(gAg) # (num_samples,)
    #     lcb_a = f.ravel() - algo.hparams.beta * cnf.ravel()  # (num_samples,)
    #     t5 = time.time()
    #     print(t2-t1,t3-t2, t4-t3, t5-t4)

    start = time.time()
    pred_actions = algo.sample_action(test_contexts) 
    # print(pred_actions)
    elapsed_time = time.time() - start 
    print(elapsed_time)


if __name__ == '__main__':
    # test_LinLCB()
    # test_KernLCB()
    # test_NeuralLinLCB()
    # test_action_convolution()
    # test_NeuralBanditModelV2()
    # test_inv_sherman_morrison()
    # test_ApproxNeuraLCBV2()
    test_sample_offline_policy()