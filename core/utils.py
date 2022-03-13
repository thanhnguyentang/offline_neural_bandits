"""Define ultility functions used by multiple algorithms. """

import jax 
import jax.numpy as jnp 
import numpy as np 
from tqdm import tqdm
from easydict import EasyDict as edict
from algorithms.lin_ucb import LinUCB # for online data collection

@jax.jit 
def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

    (A + u u^T)^{-1} = A^{-1} - A^{-1} u u^T A^{-1} / (1 + u^T A^{-1} u)
    
    Args:
        A_inv: (num_actions, p, p) 
        u: (num_actions, p)
    """
    return jax.vmap(inv_sherman_morrison_single_sample)(u, A_inv)


@jax.jit
def inv_sherman_morrison_single_sample(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

    (A + u u^T)^{-1} = A^{-1} - A^{-1} u u^T A^{-1} / (1 + u^T A^{-1} u)
    
    Args:
        A_inv: (p, p) 
        u: (p,)

    Return:
        (p,p)
    """
    assert len(A_inv.shape) == 2 # ensure 2-D tensor 
    u = u.reshape(-1,1) # (p,1)

    Au = A_inv @ u # (p,1) 

    return A_inv - (Au @ Au.T) / (1 +  u.T @ Au)  


@jax.jit 
@jax.vmap 
def vectorize_tree(params):
    return jnp.hstack(( jnp.ravel(param) for param in jax.tree_leaves(params)))


def action_convolution(contexts, actions, num_actions):
    """Compute action-convoluted (one-hot) contexts. 
    
    Args:
        contexts: (None, context_dim)
        actions: (None,)

    Return:
        convoluted_contexts: (None, context_dim * num_actions)
    """
    one_hot_actions = jax.nn.one_hot(actions, num_actions) # (None, num_actions) 
    convoluted_contexts = jax.vmap(jax.jit(jnp.kron))(one_hot_actions, contexts) # (None, context_dim * num_actions) 
    return convoluted_contexts


# action_convolution =  jax.jit(action_convolution_impure_fn, static_argnums=(2,))


def sample_offline_policy(mean_rewards, num_contexts, num_actions, pi='eps-greedy', eps=0.1, subset_r = 0.5, 
                contexts=None, rewards=None): 
    """Sample offline actions 

    Args:
        mean_rewards: (num_contexts, num_actions)
        num_contexts: int 
        num_actions: int
        pi: ['eps-greedy', 'subset', 'online']
    """
    if pi == 'subset':
        subset_s = int(num_actions * subset_r)
        subset_mean_rewards = mean_rewards[np.arange(num_contexts), :subset_s]
        actions = np.argmax(subset_mean_rewards, axis=1)
        return actions 
    elif pi == 'eps-greedy':
        uniform_actions = np.random.randint(low=0, high=num_actions, size=(num_contexts,)) 
        opt_actions = np.argmax(mean_rewards, axis=1)
        delta = np.random.uniform(size=(num_contexts,))
        selector = np.array(delta <= eps).astype('float32') 
        actions = selector.ravel() * uniform_actions + (1 - selector.ravel()) * opt_actions 
        actions = actions.astype('int')
        return actions
    elif pi == 'online':
        # Create offline data that is dependent on the past data
        assert contexts is not None 
        assert rewards is not None
        hparams = edict({
            'context_dim': contexts.shape[1], 
            'num_actions': num_actions, 
            'beta': 0.1, 
            'lambd0': 0.1, 
        })

        opt_actions = np.argmax(mean_rewards, axis=1)
        delta = np.random.uniform(size=(num_contexts,))
        selector = np.array(delta <= eps).astype('float32') 

        algo = LinUCB(hparams) 

        algo.reset(1111)
        actions = []
        for i in tqdm(range(num_contexts),ncols=75):
            c = contexts[i:i+1,:]
            a_onl = algo.sample_action(c)
            # Combine a_onl and a_opt to make sure the offline data has a good coverage of the optimal policy
            a = selector[i] * a_onl + (1-selector[i]) * opt_actions[i:i+1]
            a = a.astype('int')
            r = rewards[i:i+1,a[0]:a[0]+1]  
            algo.update(c,a,r)
            actions.append(a[0])
        return np.array(actions).astype('int')
    else:
        raise NotImplementedError('{} is not implemented'.format(pi))

def mixed_policy(num_actions, exp_reward, p_opt, p_uni):
    num_contexts = exp_reward.shape[0]

    opt_action = np.argmax(exp_reward, axis=-1).reshape(-1,1) 
    uni_action = np.random.randint(0, num_actions, size=(num_contexts, 1))
    subopt_action = ( opt_action + np.random.randint(1, num_actions, size=(num_contexts, 1)) ) % num_actions 
    sel = np.random.uniform(size=(num_contexts,1)) 
    sel_opt = np.asarray(sel < p_opt, dtype='float32')
    sel_uni = np.asarray(1 - sel < p_uni, dtype='float32')
    sel_non = 1 - sel_opt - sel_uni 
    off_action = sel_opt * opt_action + sel_uni * uni_action + sel_non * subopt_action
    
    return off_action.astype('int32')