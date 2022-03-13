"""Define synthetic data sampler. """
from tqdm import tqdm
import numpy as np 
import os
from core.utils import sample_offline_policy, mixed_policy

if __name__ == '__main__':
    print('WARNING: This is to generate meta data for dataset generation, and should only be performed once. Quit now if you are not sure what you are doing!!!')
    s = input('Type yesimnotstupid to proceed: ')
    if s == 'yesimnotstupid':
        if not os.path.exists('data/syn_meta'):
            os.makedirs('data/syn_meta')

        for sim_id in range(10):
            np.random.seed(sim_id)
            indices = np.random.permutation(100000)
            np.save('data/syn_meta/indices_{}.npy'.format(sim_id), indices)

            np.random.seed(sim_id)
            test_indices = np.random.permutation(100000)
            np.save('data/syn_meta/test_indices_{}.npy'.format(sim_id), test_indices)

            np.random.seed(sim_id)
            context_dim = 20
            num_actions = 30

            # Generate big arrays of contexts 
            contexts = np.random.uniform(-1,1, size=(100000, context_dim))
            contexts /= np.linalg.norm(contexts, axis=1)[:,None]
            np.save('data/syn_meta/contexts_{}.npy'.format(sim_id), contexts) 

            test_contexts = np.random.uniform(-1,1, size=(100000, context_dim))
            test_contexts /= np.linalg.norm(test_contexts, axis=1)[:,None]
            np.save('data/syn_meta/test_contexts_{}.npy'.format(sim_id), test_contexts) 


            ## quadratic 
            thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
            thetas /= np.linalg.norm(thetas, axis=0)[None,:]
            np.save('data/syn_meta/quadratic_thetas_{}.npy'.format(sim_id), thetas)

            ## quadratic2 
            A = np.random.randn(context_dim, context_dim, num_actions) 
            np.save('data/syn_meta/quadratic2_A_{}.npy'.format(sim_id), A)

            ## cosine  
            thetas = np.random.uniform(-1,1,size=(context_dim, num_actions))  
            thetas /= np.linalg.norm(thetas, axis=0)[None,:]
            np.save('data/syn_meta/cosine_thetas_{}.npy'.format(sim_id), thetas)


#================
# Synthetic data 
#================
class SyntheticData(object):
    def __init__(self, num_contexts, num_test_contexts, 
            context_dim = 20, 
            num_actions = 30, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5, 
            noise_std = 0.1, 
            name='quadratic'): 
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts 
        self.context_dim = context_dim
        self.num_actions = num_actions 
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.num_samples = 100000
        self.noise_std = noise_std
        self.name = name

    def reset_data(self, sim_id=0): 
        # Load meta-data to generate dataset
        indices = np.load('data/syn_meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/syn_meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]] 

        large_contexts =  np.load('data/syn_meta/contexts_{}.npy'.format(sim_id))
        large_test_contexts =  np.load('data/syn_meta/test_contexts_{}.npy'.format(sim_id))

        contexts = large_contexts[ind,:]
        test_contexts = large_test_contexts[test_ind, :]

        if self.name == 'quadratic':
            thetas = np.load('data/syn_meta/quadratic_thetas_{}.npy'.format(sim_id))
            h = lambda x: 10 * np.square(x @ thetas) # x: (None, context_dim), h: (None, num_actions)
        elif self.name == 'quadratic2':
            A = np.load('data/syn_meta/quadratic2_A_{}.npy'.format(sim_id))
            h = lambda x: np.sum(np.square(x @ A), axis=0) # x: (None, context_dim), h: (None, num_actions) 
        elif self.name == 'cosine':
            thetas = np.load('data/syn_meta/cosine_thetas_{}.npy'.format(sim_id))
            h = lambda x: np.cos(3 * x @ thetas) 
        else:
            raise ValueError

        mean_rewards = h(contexts)
        mean_test_rewards = h(test_contexts)
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r, contexts, rewards)

        dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards) 
        return dataset 

###======================
def sample_latent_manifold_data(num_contexts, num_test_contexts, \
        num_actions, context_dim, latent_dim, radius, pi, eps, subset_r):

    ## Meta-info 
    h = lambda x: 10 * np.square(x) 

    # name = 'latent_manifold'
    # reward_type = 'deterministic' if std == 0 else 'stochastic'
    # if verbose: 
    #     print('{} | num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
    #         name, num_actions, num_contexts, context_dim, reward_type
    #     ))

    U = rvs(context_dim) # (d,d)
    U1 = U[:, :latent_dim] # (d,d0)
    U2 = U[:, latent_dim:] # (d,d-d0)

    # coeffs = [np.random.uniform(0, 1, (poly + 1, )) for _ in range(num_actions)] 

    W = []
    for _ in range(num_actions):
        # Wi = []
        # for _ in range(poly + 1):
        #     Wi.append(unif_sphere(latent_dim,1,1).ravel())
        # W.append(Wi)
        W.append(unif_sphere(latent_dim,1,1).ravel())

    ## Generate train data 
    # Generate contexts
    z1 = unif_sphere(num_contexts, latent_dim, radius * np.sqrt(latent_dim)) # (n,d0)
    z2 = unif_sphere(num_contexts, context_dim - latent_dim, np.sqrt(context_dim - latent_dim)) # (n,d - d0)
    X = z1 @ U1.T + z2 @ U2.T # (n,d)
    X /= np.linalg.norm(X, axis=1)[:,None]

    # Generate mean rewards 
    B = X @ U1 # (n,d0)  
    mean_rewards = [] 
    for i in range(num_actions):
        # R = 0 
        # for j in range(poly + 1):
        #     R += coeffs[i][j] * np.power(B @ W[i][j], j).ravel() # (n,) 
        # mean_rewards.append(R) 
        mean_rewards.append( h(B @ W[i]).ravel() )
    mean_rewards = np.array(mean_rewards).T 
    actions = sample_offline_policy(mean_rewards, num_contexts, num_actions, pi, eps, subset_r)

    ## Generate test data 
    z1 = unif_sphere(num_test_contexts, latent_dim, radius * np.sqrt(latent_dim)) # (n,d0)
    z2 = unif_sphere(num_test_contexts, context_dim - latent_dim, np.sqrt(context_dim - latent_dim)) # (n,d - d0)
    X_test = z1 @ U1.T + z2 @ U2.T # (n,d)
    X_test /= np.linalg.norm(X_test, axis=1)[:,None]

    B_test = X_test @ U1 # (n,d0)  
    test_mean_rewards = [] 
    for i in range(num_actions):
        # R = 0 
        # for j in range(poly + 1):
        #     R += coeffs[i][j] * np.power(B_test @ W[i][j], j).ravel() # (n,)
        # test_mean_rewards.append(R) 
        test_mean_rewards.append( h(B_test @ W[i]).ravel() )
    test_mean_rewards = np.array(test_mean_rewards).T 

    dataset = (X, actions, mean_rewards, X_test, test_mean_rewards) 

    return dataset

#=========================================#
def sample_linear_data(num_contexts, num_test_contexts, context_dim, num_actions, sigma=0.0, p_opt=0.1, p_uni=0.1, verbose=True):
    """Samples data from linearly parameterized arms.
    The reward for context X and arm j is given by X^T beta_j, for some latent
    set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly
    at random, the contexts are Gaussian, and sigma-noise is added to the rewards.
    Args:
        num_contexts: Number of contexts to sample.
        dim_context: Dimension of the contexts.
        num_actions: Number of arms for the multi-armed bandit.
        sigma: Standard deviation of the additive noise. Set to zero for no noise.
    Returns:
        data: A [n, d+k] numpy array with the data.
        betas: Latent parameters that determine expected reward for each arm.
        opt: (optimal_rewards, optimal_actions) for all contexts.
    """
    ## Meta-info 
    name = 'linear'
    reward_type = 'deterministic' if sigma == 0 else 'stochastic'
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_contexts, context_dim, reward_type
        ))

    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)

    # Generate train data 
    contexts = np.random.normal(size=[num_contexts, context_dim])
    mean_rewards = np.dot(contexts, betas) # (num_contexts, num_actions)
    rewards = mean_rewards + np.random.normal(scale=sigma, size=mean_rewards.shape)

    off_action = mixed_policy(num_actions, mean_rewards, p_opt, p_uni)
    rewards = rewards[np.arange(num_contexts), off_action.ravel()].reshape(-1,1)
    dataset = (contexts, off_action, rewards) 

    # Generate test data 
    test_contexts = np.random.normal(size=[num_test_contexts, context_dim])
    test_mean_rewards = np.square(np.dot(test_contexts, betas)) # (num_contexts, num_actions)
    test_dataset = (test_contexts, test_mean_rewards) 

    return dataset, test_dataset


def sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v,
                             mu_large, std_large, p_opt=0.1, p_uni=0.1, verbose=True):
    """Samples from Wheel bandit game (see https://arxiv.org/abs/1802.09127).
    Args:
        num_contexts: Number of points to sample, i.e. (context, action rewards).
        delta: Exploration parameter: high reward in one region if norm above delta.
        mean_v: Mean reward for each action if context norm is below delta.
        std_v: Gaussian reward std for each action if context norm is below delta.
        mu_large: Mean reward for optimal action if context norm is above delta.
        std_large: Reward std for optimal action if context norm is above delta.
    Returns:
        dataset: Sampled matrix with n rows: (context, action rewards).
        opt_vals: Vector of expected optimal (reward, action) for each context.
    """

    context_dim = 2
    num_actions = 5

    ## Meta-info 
    name = 'wheel-bandit'
    reward_type = 'stochastic'
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_contexts, context_dim, reward_type
        ))


    data = []
    

    # sample uniform contexts in unit ball
    while len(data) < num_contexts:
        raw_data = np.random.uniform(-1, 1, (int(num_contexts / 3), context_dim))

        for i in range(raw_data.shape[0]):
            if np.linalg.norm(raw_data[i, :]) <= 1: # reject points beyond the unit ball 
                data.append(raw_data[i, :])

    contexts = np.stack(data)[:num_contexts, :]

    # sample rewards
    rewards = [] 
    mean_rewards = np.ones((num_contexts, num_actions)) * np.min(mean_v)
    mean_rewards[:,-1] = np.max(mean_v)
    for i in range(num_contexts):
        r = [np.random.normal(mean_v[j], std_v[j]) for j in range(num_actions)]
        if np.linalg.norm(contexts[i, :]) >= delta:
            # large reward in the right region for the context
            r_big = np.random.normal(mu_large, std_large)
            if contexts[i, 0] > 0:
                if contexts[i, 1] > 0:
                    r[0] = r_big
                    mean_rewards[i,0] = mu_large
                else:
                    r[1] = r_big
                    mean_rewards[i,1] = mu_large
            else:
                if contexts[i, 1] > 0:
                    r[2] = r_big
                    mean_rewards[i,2] = mu_large
                else:
                    r[3] = r_big
                    mean_rewards[i,3] = mu_large

        rewards.append(r)

    rewards = np.stack(rewards)
    off_action = mixed_policy(num_actions, mean_rewards, p_opt, p_uni)
    return contexts, rewards, off_action, mean_rewards

def rvs(dim=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in tqdm(range(1, dim)):
        x = np.random.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

def unif_sphere(n,d,r):
    """Uniformly sample `n` points from a `d`-dim sphere of radius `r` 
    """
    X = np.random.uniform(-1, 1 + 1e-6, (n,d)) 
    return r * X / np.sqrt(np.sum(np.square(X), axis=1))[:,None]
