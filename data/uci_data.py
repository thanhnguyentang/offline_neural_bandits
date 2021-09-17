

import numpy as np 
import pandas as pd 
from core.utils import sample_offline_policy
# import tensorflow as tf

if __name__ == '__main__':
    # Generate meta data, do not run it
    print('WARNING: This is to generate meta data for dataset generation, and should only be performed once. Quit now if you are not sure what you are doing!!!')
    s = input('Type yesimnotstupid to proceed: ')
    if s == 'yesimnotstupid':
        for sim_id in range(10):
            np.random.seed(sim_id)
            indices = np.random.permutation(100000)
            np.save('data/meta/indices_{}.npy'.format(sim_id), indices)

            np.random.seed(sim_id)
            test_indices = np.random.permutation(100000)
            np.save('data/meta/test_indices_{}.npy'.format(sim_id), test_indices)

            np.random.seed(sim_id)
            context_dim = 21
            num_actions = 8
            betas = np.random.uniform(-1, 1, (context_dim, num_actions))
            betas /= np.linalg.norm(betas, axis=0)
            np.save('data/meta/betas_{}.npy'.format(sim_id), betas)


def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df

class MushroomData(object):
    def __init__(self, 
                num_contexts, 
                num_test_contexts, 
                num_actions = 2, 
                r_noeat=0,
                r_eat_safe=5,
                r_eat_poison_bad=-35,
                r_eat_poison_good=5,
                prob_poison_bad=0.5,
                pi = 'eps-greedy', 
                eps = 0.1, 
                subset_r = 0.5
                ): 
        filename = 'data/mushroom.data'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions 

        self.r_noeat = r_noeat 
        self.r_eat_safe = r_eat_safe 
        self.r_eat_poison_bad = r_eat_poison_bad 
        self.r_eat_poison_good = r_eat_poison_good 
        self.prob_poison_bad = prob_poison_bad 
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r

        df = pd.read_csv(filename, header=None)
        self.df = one_hot(df, df.columns)

        print('Mushroom: num_samples: {}'.format(self.num_samples)) 

    @property 
    def num_samples(self): 
        return self.df.shape[0]


    def reset_data(self, sim_id=0): 
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            self.ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            self.ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts = self.df.iloc[self.ind, 2:].values 
        test_contexts = self.df.iloc[test_ind, 2:].values 


        # Compute mean rewards 
        no_eat_reward = self.r_noeat * np.ones((self.num_contexts, 1))
        mean_eat_poison_reward = self.r_eat_poison_bad * self.prob_poison_bad + self.r_eat_poison_good * (1 - self.prob_poison_bad)
        mean_eat_reward = self.r_eat_safe * self.df.iloc[self.ind, 0] + np.multiply(mean_eat_poison_reward, self.df.iloc[self.ind, 1])
        mean_eat_reward = mean_eat_reward.values.reshape((self.num_contexts, 1))
        mean_rewards = np.hstack((no_eat_reward, mean_eat_reward))

        no_eat_reward = self.r_noeat * np.ones((self.num_test_contexts, 1))
        mean_eat_reward = self.r_eat_safe * self.df.iloc[test_ind, 0] + np.multiply(mean_eat_poison_reward, self.df.iloc[test_ind, 1])
        mean_eat_reward = mean_eat_reward.values.reshape((self.num_test_contexts, 1))
        mean_test_rewards = np.hstack((no_eat_reward, mean_eat_reward))

        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        dataset = (contexts, actions, mean_rewards, test_contexts, mean_test_rewards) 
        return dataset 

    def reset_rewards(self): 
        no_eat_reward = self.r_noeat * np.ones((self.num_contexts, 1))
        random_poison = np.random.choice([self.r_eat_poison_bad, self.r_eat_poison_good],
                                p=[self.prob_poison_bad, 1 - self.prob_poison_bad], size= self.num_contexts)
        eat_reward = self.r_eat_safe * self.df.iloc[self.ind, 0]
        eat_reward += np.multiply(random_poison, self.df.iloc[self.ind, 1])
        eat_reward = eat_reward.values.reshape((self.num_contexts, 1))

        return np.hstack((no_eat_reward, eat_reward))

class JesterData(object):
    def __init__(self,
            num_contexts,
            num_test_contexts, 
            context_dim = 32, 
            num_actions = 8, 
            noise_std = 0.1, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5
        ): 
        file_name = 'data/jester.npy'
        with open(file_name, 'rb') as f:
            self.dataset = np.load(f)

        assert context_dim + num_actions == self.dataset.shape[1]

        # if shuffle_cols:
        #     dataset = dataset[:, np.random.permutation(dataset.shape[1])]
        # if shuffle_rows:
        #     np.random.shuffle(dataset)

        ## Meta-info 
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.noise_std = noise_std 
        reward_type = 'stochastic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            'jester', num_actions, self.dataset.shape[0], context_dim, reward_type
        ))

    @property
    def num_samples(self):
        return self.dataset.shape[0]

    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            self.ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            self.ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts = self.dataset[self.ind, :self.context_dim]
        mean_rewards = self.dataset[self.ind, self.context_dim:] 

        test_contexts = self.dataset[test_ind, :self.context_dim]
        mean_test_rewards = self.dataset[test_ind, self.context_dim:] 

        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        dataset = (contexts, actions, mean_rewards, test_contexts, mean_test_rewards) 
        return dataset 

    def reset_rewards(self): 
        mean_rewards = self.dataset[self.ind, self.context_dim:]
        return mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)

class StatlogData(object):
    def __init__(self,
            num_contexts, 
            num_test_contexts, 
            context_dim = 9, 
            num_actions = 7, 
            noise_std = 0, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5,
            remove_underrepresented=False, 
            shuffle_rows=True
        ):
        file_name = 'data/shuttle.trn'
        with open(file_name, 'r') as f:
            dataset = np.loadtxt(f)

        ## Meta-info 
        name = 'statlog'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.noise_std = noise_std

        if shuffle_rows:
            np.random.shuffle(dataset)

        contexts = dataset[: :-1]
        labels = dataset[:, -1].astype(int) - 1
        if remove_underrepresented:
            contexts, labels = remove_underrepresented_classes(contexts, labels)
        self.contexts, self.mean_rewards = classification_to_bandit_problem(contexts, labels, num_actions)

        reward_type = 'deterministic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, self.num_samples, context_dim, reward_type
        ))

    
    @property
    def num_samples(self):
        return self.contexts.shape[0]

    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            self.ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            self.ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts, mean_rewards = self.contexts[ind,:], self.mean_rewards[ind,:] 
        test_contexts, mean_test_rewards = self.contexts[test_ind,:], self.mean_rewards[test_ind,:]

        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        dataset = (contexts, actions, mean_rewards, test_contexts, mean_test_rewards) 
        return dataset 

class CoverTypeData(object): 
    def __init__(self,
            num_contexts, 
            num_test_contexts, 
            num_actions = 7, 
            context_dim = 54,
            noise_std = 0, 
            pi = 'eps-greedy', 
            eps = 0.1, 
            subset_r = 0.5,
            remove_underrepresented=False, 
            shuffle_rows=True
        ):
        file_name = 'data/covtype.data'
        with open(file_name, 'r') as f:
            df = pd.read_csv(f, header=0, na_values=['?']).dropna()

        name = 'covertype'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.pi = pi 
        self.eps = eps 
        self.subset_r = subset_r
        self.noise_std = noise_std

            
        if shuffle_rows:
            df = df.sample(frac=1)
        # data = df.iloc[:num_contexts, :]

        # Assuming what the paper calls response variable is the label?
        # Last column is label.
        labels = df[df.columns[-1]].astype('category').cat.codes.to_numpy()
        df = df.drop([df.columns[-1]], axis=1)

        if remove_underrepresented:
            df, labels = remove_underrepresented_classes(df, labels)

        contexts = df.to_numpy()
        self.contexts, self.rewards = classification_to_bandit_problem(contexts, labels, num_actions)
        
        reward_type = 'deterministic'
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, self.num_samples, context_dim, reward_type
        ))

    def reset_data(self, sim_id=0):
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)

        # Generate inds
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            self.ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            self.ind = indices[i[:self.num_contexts]]

        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]


        contexts = self.contexts[self.ind,:] 
        mean_rewards = self.rewards[self.ind,:] 
        test_contexts = self.contexts[test_ind,:]
        mean_test_rewards = self.rewards[test_ind,:]
        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        dataset = (contexts, actions, mean_rewards, test_contexts, mean_test_rewards) 
        return dataset 

    @property 
    def num_samples(self):
        return self.contexts.shape[0]



class StockData(object):
    def __init__(self, 
                num_contexts, 
                num_test_contexts, 
                num_actions = 8, 
                noise_std=0.1, 
                shuffle_rows=True,
                pi = 'eps-greedy', 
                eps = 0.1, 
                subset_r = 0.5
                ): 
        filename = 'data/raw_stock_contexts'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions 

        self.noise_std = noise_std
        self.shuffle_rows = shuffle_rows
        self.pi = pi
        self.eps = eps
        self.subset_r = subset_r


        with open(filename, 'r') as f:
            contexts = np.loadtxt(f, skiprows=1)

        if shuffle_rows:
            np.random.shuffle(contexts)

        self.contexts = contexts
        self.context_dim = contexts.shape[1]

        print('Stock: num_samples: {}'.format(self.num_samples)) 

    @property 
    def num_samples(self): 
        return len(self.contexts)


    def reset_data(self, sim_id=0): 
        # Load meta-data to generate dataset
        indices = np.load('data/meta/indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        test_indices = np.load('data/meta/test_indices_{}.npy'.format(sim_id)) # random permutation of np.arange(100000)
        self.betas = np.load('data/meta/betas_{}.npy'.format(sim_id))

        # Generate inds 
        indices = indices % self.num_samples
        test_indices = test_indices % self.num_samples

        if self.num_contexts > self.num_samples:
            self.ind = indices[:self.num_contexts]
        else:
            # then select self.num_contexts first distinc elements of indices
            i = np.unique(indices,return_index=True)[1]
            i.sort()
            self.ind = indices[i[:self.num_contexts]]
        
        if self.num_test_contexts > self.num_samples:
            test_ind = test_indices[:self.num_test_contexts]
        else:
            i = np.unique(test_indices,return_index=True)[1]
            i.sort()
            test_ind = test_indices[i[:self.num_test_contexts]]

        contexts = self.contexts[self.ind,:] 
        test_contexts = self.contexts[test_ind,:]
        # Compute rewards
        mean_rewards = np.dot(contexts, self.betas) # (num_contexts, num_actions)
        mean_test_rewards = np.dot(test_contexts, self.betas) # (num_contexts, num_actions)

        actions = sample_offline_policy(mean_rewards, self.num_contexts, self.num_actions, self.pi, self.eps, self.subset_r)
        dataset = (contexts, actions, mean_rewards, test_contexts, mean_test_rewards) 
        return dataset 


    def reset_rewards(self):
        contexts = self.contexts[self.ind]

        # Compute rewards
        mean_rewards = np.dot(contexts, self.betas) # (num_contexts, num_actions)
        noise = np.random.normal(scale=self.noise_std, size=mean_rewards.shape)
        rewards = mean_rewards + noise
    
        return rewards



#==================================
#==================================
def sample_mushroom_data(file_name,
                    num_contexts,
                    r_noeat=0,
                    r_eat_safe=5,
                    r_eat_poison_bad=-35,
                    r_eat_poison_good=5,
                    prob_poison_bad=0.5,
                    p_opt = 0, 
                    p_uni = 1, 
                    verbose=True):
    """Samples bandit game from Mushroom UCI Dataset.
    Args:
        file_name: Route of file containing the original Mushroom UCI dataset.
        num_contexts: Number of points to sample, i.e. (context, action, rewards).
        r_noeat: Reward for not eating a mushroom.
        r_eat_safe: Reward for eating a non-poisonous mushroom.
        r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
        r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
        prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.
        p_opt: Probability of selecting an optimal action in the offline data. 
        p_uni: Probability of selecting an uniformly generated action in the offline data.
    Returns:
        contexts: An array of sampled contexts. 
        reward: A matrix of random reward for no_eat and eat (no_eat = 0, eat = 1)
        off_action: A matrix of offline actions
        exp_reward: A matrix of expected reward for no_eat and eat action. 
    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
    """
    assert p_uni + p_opt <= 1 

    # first two cols of df encode whether mushroom is edible or poisonous
    num_actions = 2 # no_eat vs eat 

    df = pd.read_csv(file_name, header=None)
    df = one_hot(df, df.columns)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=False)

    contexts = df.iloc[ind, 2:].values

    if verbose:
        print('mushroom: total_samples : {} | num_actions: {} | context_dim: {} | reward_type: stochastic'.format(
            df.shape[0], num_actions, contexts.shape[1]))

    no_eat_reward = r_noeat * np.ones((num_contexts, 1))


    random_poison = np.random.choice([r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # Compute expected reward for each action 
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad + r_eat_poison_good * (1 - prob_poison_bad)
    exp_eat_reward = r_eat_safe * df.iloc[ind, 0] + np.multiply(exp_eat_poison_reward, df.iloc[ind, 1])
    exp_eat_reward = exp_eat_reward.values.reshape((num_contexts, 1))

    mean_rewards = np.hstack((no_eat_reward, exp_eat_reward))

    # Generate offline actions 
    actions = sample_offline_policy(mean_rewards, num_contexts, num_actions, pi, eps, subset_r)

    dataset = (contexts, actions, mean_rewards, test_contexts, test_mean_rewards) 

    return contexts, np.hstack((no_eat_reward, eat_reward)), off_action, exp_reward

def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
                        remove_underrepresented=False, p_opt=0.1, p_uni=0.1, verbose=True):
    """Returns bandit problem dataset based on the UCI statlog data.
    Args:
        file_name: Route of file containing the Statlog dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.
    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
    https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    """

    with open(file_name, 'r') as f:
        data = np.loadtxt(f)

    ## Meta-info 
    name = 'statlog'
    num_actions = 7  # some of the actions are very rarely optimal.
    num_samples = data.shape[0] 
    context_dim = 9
    reward_type = 'deterministic' 
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_samples, context_dim, reward_type
        ))

    # Shuffle data
    if shuffle_rows:
        np.random.shuffle(data)
    data = data[:num_contexts, :]

    # Last column is label, rest are features
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int) - 1  # convert to 0 based index

    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)

    contexts, rewards = classification_to_bandit_problem(contexts, labels, num_actions)
    
    exp_rewards = rewards 
    # Generate offline actions 
    off_action = mixed_policy(num_actions, exp_rewards, p_opt, p_uni)

    return contexts, rewards, off_action, exp_rewards

def sample_adult_data(file_name, num_contexts, shuffle_rows=True,
                      remove_underrepresented=False, p_opt=0.1, p_uni=0.1, verbose=True):
    """Returns bandit problem dataset based on the UCI adult data.
    Args:
        file_name: Route of file containing the Adult dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.
    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
        * drop rows with missing values
        * convert categorical variables to 1 hot encoding
    https://archive.ics.uci.edu/ml/datasets/census+income
    """
    with open(file_name, 'r') as f:
        df = pd.read_csv(f, header=None,
                        na_values=[' ?']).dropna()
    num_samples = df.shape[0] 

    if shuffle_rows:
        df = df.sample(frac=1)
    # df = df.iloc[:num_contexts, :]

    labels = df[6].astype('category').cat.codes.to_numpy()
    df = df.drop([6], axis=1)

    # Convert categorical variables to 1 hot encoding
    cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
    df = pd.get_dummies(df, columns=cols_to_transform)

    if remove_underrepresented:
        df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.to_numpy()

    ## Meta-info 
    name = 'adult'
    num_actions = 14  # some of the actions are very rarely optimal.
    context_dim = contexts.shape[1]
    reward_type = 'deterministic' 
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_samples, context_dim, reward_type
        ))

    contexts, rewards = classification_to_bandit_problem(contexts[:num_contexts], labels[:num_contexts], num_actions)
    exp_rewards = rewards 
    # Generate offline actions 
    off_action = mixed_policy(num_actions, exp_rewards, p_opt, p_uni)

    return contexts, rewards, off_action, exp_rewards

def sample_stock_data(file_name, num_contexts,
                      sigma=0.1, shuffle_rows=True, p_opt=0.1, p_uni=0.1, verbose=True):
    """Samples linear bandit game from stock prices dataset.
    Args:
        file_name: Route of file containing the stock prices dataset.
        context_dim: Context dimension (i.e. vector with the price of each stock).
        num_actions: Number of actions (different linear portfolio strategies).
        num_contexts: Number of contexts to sample.
        sigma: Vector with additive noise levels for each action.
        shuffle_rows: If True, rows from original dataset are shuffled.
    Returns:
        dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).
        opt_vals: Vector of expected optimal (reward, action) for each context.
    """

    with open(file_name, 'r') as f:
        contexts = np.loadtxt(f, skiprows=1)

    if shuffle_rows:
        np.random.shuffle(contexts)
    contexts = contexts[:num_contexts, :]

    ## Meta-info 
    name = 'stock'
    reward_type = 'stochastic' 
    context_dim = contexts.shape[1]
    num_actions = 8
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, contexts.shape[0], context_dim, reward_type
        ))

    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)

    mean_rewards = np.dot(contexts, betas) # (num_contexts, num_actions)
    noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
    rewards = mean_rewards + noise

    opt_actions = np.argmax(mean_rewards, axis=1)
    off_action = mixed_policy(num_actions, mean_rewards, p_opt, p_uni)
    # opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
    # return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)
    return contexts, rewards, off_action, mean_rewards

def sample_jester_data(file_name, num_contexts,
                       shuffle_rows=True, shuffle_cols=False, p_opt=0.1, p_uni=0.1, verbose=True):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
        file_name: Route of file containing the modified Jester dataset.
        context_dim: Context dimension (i.e. vector with some ratings from a user).
        num_actions: Number of actions (number of joke ratings to predict).
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
        dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """

    with open(file_name, 'rb') as f:
        dataset = np.load(f)

    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)

    ## Meta-info 
    name = 'jester'
    reward_type = 'deterministic' 
    context_dim = 32
    num_actions = 8
    num_samples = dataset.shape[0]
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_samples, context_dim, reward_type
        ))

    contexts = dataset[:num_contexts, :context_dim]
    mean_rewards = dataset[:num_contexts, context_dim:] 

    opt_actions = np.argmax(mean_rewards, axis=1)
    off_action = mixed_policy(num_actions, mean_rewards, p_opt, p_uni)
    return contexts, mean_rewards, off_action, mean_rewards

def sample_census_data(file_name, num_contexts, shuffle_rows=True,
                       remove_underrepresented=False, p_opt=0.1, p_uni=0.1, verbose=True):
    """Returns bandit problem dataset based on the UCI census data.
    Args:
        file_name: Route of file containing the Census dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.
    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
        * drop rows with missing labels
        * convert categorical variables to 1 hot encoding
    Note: this is the processed (not the 'raw') dataset. It contains a subset
    of the raw features and they've all been discretized.
    https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
    """
    # Note: this dataset is quite large. It will be slow to load and preprocess.
    with open(file_name, 'r') as f:
        df = (pd.read_csv(f, header=0, na_values=['?'])
            .dropna())


    ## Meta-info 
    name = 'census'
    num_actions = 9  # some of the actions are very rarely optimal.
    num_samples = df.shape[0] 
    context_dim = 389 
    reward_type = 'deterministic' 
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_samples, context_dim, reward_type
        ))

    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]

    # Assuming what the paper calls response variable is the label?
    labels = df['dOccup'].astype('category').cat.codes.to_numpy()
    # print(labels)
    # In addition to label, also drop the (unique?) key.
    df = df.drop(['dOccup', 'caseid'], axis=1)

    # All columns are categorical. Convert to 1 hot encoding.
    df = pd.get_dummies(df, columns=df.columns)

    if remove_underrepresented:
        df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.to_numpy()

    contexts, rewards = classification_to_bandit_problem(contexts, labels, num_actions)
    exp_rewards = rewards 
    off_action = mixed_policy(num_actions, exp_rewards, p_opt, p_uni)
    return contexts, rewards, off_action, exp_rewards

def sample_covertype_data(file_name, num_contexts, shuffle_rows=True,
                          remove_underrepresented=False, p_opt=0.1, p_uni=0.1, verbose=True):
    """Returns bandit problem dataset based on the UCI Cover_Type data.
    Args:
        file_name: Route of file containing the Covertype dataset.
        num_contexts: Number of contexts to sample.
        shuffle_rows: If True, rows from original dataset are shuffled.
        remove_underrepresented: If True, removes arms with very few rewards.
    Returns:
        dataset: Sampled matrix with rows: (context, action rewards).
        opt_vals: Vector of deterministic optimal (reward, action) for each context.
    Preprocessing:
        * drop rows with missing labels
        * convert categorical variables to 1 hot encoding
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """
    with open(file_name, 'r') as f:
        df = (pd.read_csv(f, header=0, na_values=['?'])
            .dropna())

     ## Meta-info 
    name = 'covertype'
    num_actions = 7  # some of the actions are very rarely optimal.
    num_samples = df.shape[0] 
    context_dim = 54 
    reward_type = 'deterministic' 
    if verbose: 
        print('{}: num_actions: {} | total_samples: {} | context_dim: {} | reward_type: {}'.format(
            name, num_actions, num_samples, context_dim, reward_type
        ))


    if shuffle_rows:
        df = df.sample(frac=1)
    df = df.iloc[:num_contexts, :]

    # Assuming what the paper calls response variable is the label?
    # Last column is label.
    labels = df[df.columns[-1]].astype('category').cat.codes.to_numpy()
    df = df.drop([df.columns[-1]], axis=1)

    # All columns are either quantitative or already converted to 1 hot.
    if remove_underrepresented:
        df, labels = remove_underrepresented_classes(df, labels)
    contexts = df.to_numpy()

    contexts, rewards = classification_to_bandit_problem(contexts, labels, num_actions)
    exp_rewards = rewards 
    off_action = mixed_policy(num_actions, exp_rewards, p_opt, p_uni)
    return contexts, rewards, off_action, exp_rewards

def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Normalize contexts and encode deterministic rewards."""

    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # Normalize features
    contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

    # One hot encode labels as rewards
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    return contexts, rewards #, (np.ones(num_contexts), labels)

def safe_std(values):
    """Remove zero std values for ones."""
    return np.array([val if val != 0.0 else 1.0 for val in values])

def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Removes classes when number of datapoints fraction is below a threshold."""

    # Threshold doesn't seem to agree with https://arxiv.org/pdf/1706.04687.pdf
    # Example: for Covertype, they report 4 classes after filtering, we get 7?
    total_count = labels.shape[0]
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return features[keep], labels[np.array(keep)]

# file_name = 'mushroom.data'
# contexts, rewards, off_action, exp_rewards = sample_mushroom_data(file_name, num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'shuttle.trn'
# contexts, rewards, off_action, exp_rewards = sample_statlog_data(file_name, num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'adult.data'
# contexts, rewards, off_action, exp_rewards = sample_adult_data(file_name, num_contexts=30, p_opt=1, p_uni=0)

# file_name = 'raw_stock_contexts'
# contexts, rewards, off_action, exp_rewards = sample_stock_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'jester.npy'
# contexts, rewards, off_action, exp_rewards = sample_jester_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'USCensus1990.data.txt'
# contexts, rewards, off_action, exp_rewards = sample_census_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# file_name = 'covtype.data'
# contexts, rewards, off_action, exp_rewards = sample_covertype_data(file_name, 
#     num_contexts=10000, p_opt=1, p_uni=0)

# print(contexts.shape) 
# print(rewards.shape)
# print(exp_rewards.shape)
# print(off_action.shape)





# print(contexts)
