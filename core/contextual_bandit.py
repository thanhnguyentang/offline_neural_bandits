"""Define an contextual bandit class. """

import numpy as np
from tqdm import tqdm
from timeit import timeit 

def action_stats(actions, num_actions):
    """Compute the freq of each action.

    Args:
        actions: (None, )
        num_actions: int 
    """

    stats = [] 
    for a in range(num_actions):
        stats.append(np.mean(np.asarray(actions == a).astype('float32')))
    return stats
    
def action_accuracy(pred_actions, opt_actions):
    """Compute accuracy between predicted actions and optimal actions. 

    Args:
        pred_actions: (None,)
        opt_actions: (None,)
    """
    return np.mean(np.asarray(pred_actions == opt_actions).astype('float32'))

def run_contextual_bandit(dataset, test_dataset, algos, test_freq, verbose, debug, normalize=False):
    """Run an offline contextual bandit problem on a set of algorithms in the same dataset. 

    Args:
        dataset: A tuple of contexts, offline actions, and rewards of the chosen actions.
        test_dataset: A tuple of test contexts and expected reward values. 
        algs: A list of algorithms to run on the bandit instance.  
    """

    num_contexts = dataset[0].shape[0]
    num_test_contexts = test_dataset[0].shape[0]
    num_actions = test_dataset[1].shape[1]

    # Create a bandit instance 
    cmab = ContextualBandit() 
    cmab.feed_data(*dataset) 
    # print(dataset[0].shape, dataset[1].shape, dataset[2].shape)
    cmab.feed_test_data(*test_dataset) 

    # Run the offline contextual bandit in an online manner 
    subopts = [
        [] for _ in range(len(algos))
    ]

    act_errs = [
        [] for _ in range(len(algos))
    ]

    # Compute test values and opt actions 
    opt_val = np.max(cmab.test_vals, axis=1) 
    opt_act = np.argmax(cmab.test_vals, axis=1) 

    for i in tqdm(range(num_contexts)):
        context = cmab.context(i) 
        action = cmab.action(i) 
        reward = cmab.reward(i) # reward for the chosen action 

        for j,a in enumerate(algos): 
            # Add data and update the internal state of each offline algorithm 
            a.update(context, action, reward) 

            # Test alg 
            if i % test_freq == 0:
                test_action = a.sample_action(cmab.test_contexts) 
                sel_val = cmab.test_vals[np.arange(num_test_contexts), test_action.ravel()]
                if normalize:
                    test_subopt = np.mean(1 - sel_val / opt_val) 
                else:
                    test_subopt = np.mean(opt_val - sel_val)
                action_acc = action_accuracy(test_action.ravel(), opt_act.ravel()) 

                if verbose: 
                    
                    print('{} | regret: {} | acc: {} | iter: {}/{}'.format(a.name, test_subopt, action_acc, i, num_contexts))
                    if debug: 
                        sel_stats = action_stats(test_action.ravel(), num_actions) 
                        opt_stats = action_stats(opt_act.ravel(), num_actions)
                        print('     opt_rate: {} | pred_rate: {}'.format(opt_stats, sel_stats))
                        a.monitor(context, action, reward)

                subopts[j].append(test_subopt) 
                act_errs[j].append(1 - action_acc) 

        if i % test_freq == 0:
            if verbose:
                print('================')

    # Free mem
    del cmab 
    for a in algos:
        del a 

    return np.array(subopts), np.array(act_errs)
    # return np.array(subopts) 


def contextual_bandit_runner(algos, dataset, noise_std, num_sim, update_freq, test_freq, verbose, debug, normalize, save_path=None):
    """Run an offline contextual bandit problem on a set of algorithms in the same dataset. 

    Args:
        dataset: A tuple of (contexts, actions, mean_rewards, test_contexts, test_mean_rewards).
        algs: A list of algorithms to run on the bandit instance.  
    """

    # Create a bandit instance 
    cmab = OfflineContextualBandit(noise_std, *dataset) 


    regrets = [] # (num_sim, num_algos, T) 
    errs = [] # (num_sim, num_algos, T) 
    for sim in range(num_sim):
        # Run the offline contextual bandit in an online manner 
        print('Simulation: {}/{}'.format(sim + 1, num_sim))
        cmab.reset_rewards() 
        for algo in algos:
            algo.reset(sim * 1111)

        subopts = [
            [] for _ in range(len(algos))
        ]

        act_errs = [
            [] for _ in range(len(algos))
        ]

        # Compute test values and opt actions 
        opt_vals = np.max(cmab.test_mean_rewards, axis=1) 
        opt_actions = np.argmax(cmab.test_mean_rewards, axis=1) 

        for i in tqdm(range(cmab.num_contexts)):
            start_time = timeit()

            c,a,r = cmab.get_data(i) 

            for j,algo in enumerate(algos): 
                algo.update_buffer(c,a,r)
                # Add data and update the internal state of each offline algorithm 
                if i % algo.update_freq == 0 and algo.name != 'KernLCB':
                    algo.update(c, a, r) 

                # Test alg 
                if i % test_freq == 0:
                    if algo.name == 'KernLCB': 
                        algo.update()

                    if algo.name == 'KernLCB' and algo.X.shape[0] >= algo.hparams.max_num_sample:
                        print('KernLCB reuses the last 1000 points for prediction!')
                        test_subopt = subopts[j][-1] # reuse the last result
                        action_acc = 1 - act_errs[j][-1]
                        if verbose: 
                            print('[sim: {}/{} | iter: {}/{}] {} | regret: {} | acc: {} | '.format(
                                sim+1, num_sim,  i, cmab.num_contexts,
                                algo.name, test_subopt, action_acc))
                    else:
                        test_actions = algo.sample_action(cmab.test_contexts) 
                        sel_vals = cmab.test_mean_rewards[np.arange(cmab.num_test_contexts), test_actions.ravel()]
                        if normalize:
                            test_subopt = np.mean(1 - sel_vals / opt_vals) 
                        else:
                            test_subopt = np.mean(opt_vals - sel_vals)
                        action_acc = action_accuracy(test_actions.ravel(), opt_actions.ravel()) 

                        if verbose: 
                            print('[sim: {}/{} | iter: {}/{}] {} | regret: {} | acc: {} | '.format(
                                sim+1, num_sim,  i, cmab.num_contexts,
                                algo.name, test_subopt, action_acc))
                            if debug: 
                                sel_stats = action_stats(test_actions.ravel(), cmab.num_actions) 
                                opt_stats = action_stats(opt_actions.ravel(), cmab.num_actions)
                                print('     opt_rate: {} | pred_rate: {}'.format(opt_stats, sel_stats))
                                algo.monitor(c, a, r)

                    subopts[j].append(test_subopt) 
                    act_errs[j].append(1 - action_acc) 
        
            time_elapsed = timeit() - start_time
            if i % test_freq == 0:
                if verbose:
                    print('Time elapse per iteration: {}'.format(time_elapsed))
                    print('================')
                    
        regrets.append(np.array(subopts)) 
        errs.append(np.array(act_errs)) 

        if save_path: # save for every simulation
            np.savez(save_path, np.array(regrets), np.array(errs) ) 

    return np.array(regrets), np.array(errs) 

def realworld_contextual_bandit_runner(algos, dataset, data, \
            num_sim, update_freq, test_freq, verbose, debug, normalize, save_path=None):
    """Run an offline contextual bandit problem on a set of algorithms in the same dataset. 

    Args:
        dataset: A tuple of (contexts, actions, mean_rewards, test_contexts, test_mean_rewards).
        algs: A list of algorithms to run on the bandit instance.  
    """

    # Create a bandit instance 
    cmab = OfflineContextualBandit(data.noise_std, *dataset) 
    mean_rewards = dataset[2]
    regrets = [] # (num_sim, num_algos, T) 
    errs = [] # (num_sim, num_algos, T) 
    for sim in range(num_sim):
        # Run the offline contextual bandit in an online manner 
        print('Simulation: {}/{}'.format(sim + 1, num_sim))
        cmab.reset_rewards() # only for independent noise data 
        # cmab.set_rewards(rewards)

        # cmab.set_rewards(data.reset_rewards())
        for algo in algos:
            algo.reset(sim * 1111)

        subopts = [
            [] for _ in range(len(algos))
        ]

        act_errs = [
            [] for _ in range(len(algos))
        ]

        # Compute test values and opt actions 
        opt_vals = np.max(cmab.test_mean_rewards, axis=1) 
        opt_actions = np.argmax(cmab.test_mean_rewards, axis=1) 

        for i in tqdm(range(cmab.num_contexts),ncols=75):
            start_time = timeit()

            c,a,r = cmab.get_data(i) 

            for j,algo in enumerate(algos): 
                algo.update_buffer(c,a,r)
                # Add data and update the internal state of each offline algorithm 
                if i % algo.update_freq == 0 and algo.name != 'KernLCB':
                    algo.update(c, a, r) 

                # Test alg 
                if i % test_freq == 0:
                    if algo.name == 'KernLCB': 
                        algo.update()

                    if algo.name == 'KernLCB' and algo.X.shape[0] >= algo.hparams.max_num_sample:
                        print('KernLCB reuses the last 1000 points for prediction!')
                        test_subopt = subopts[j][-1] # reuse the last result
                        action_acc = 1 - act_errs[j][-1]
                        if verbose: 
                            print('[sim: {}/{} | iter: {}/{}] {} | regret: {} | acc: {} | '.format(
                                sim+1, num_sim,  i, cmab.num_contexts,
                                algo.name, test_subopt, action_acc))
                    else:
                        test_actions = algo.sample_action(cmab.test_contexts) 
                        sel_vals = cmab.test_mean_rewards[np.arange(cmab.num_test_contexts), test_actions.ravel()]
                        if normalize:
                            test_subopt = np.mean(1 - sel_vals / opt_vals) 
                        else:
                            test_subopt = np.mean(opt_vals - sel_vals)
                        action_acc = action_accuracy(test_actions.ravel(), opt_actions.ravel()) 

                        if verbose: 
                            print('[sim: {}/{} | iter: {}/{}] {} | regret: {} | acc: {} | '.format(
                                sim+1, num_sim,  i, cmab.num_contexts,
                                algo.name, test_subopt, action_acc))
                            if debug: 
                                sel_stats = action_stats(test_actions.ravel(), cmab.num_actions) 
                                opt_stats = action_stats(opt_actions.ravel(), cmab.num_actions)
                                print('     opt_rate: {} | pred_rate: {}'.format(opt_stats, sel_stats))
                                algo.monitor(c, a, r)

                    subopts[j].append(test_subopt) 
                    act_errs[j].append(1 - action_acc) 
        
            time_elapsed = timeit() - start_time
            if i % test_freq == 0:
                if verbose:
                    print('Time elapse per iteration: {}'.format(time_elapsed))
                    print('================')
                    
        regrets.append(np.array(subopts)) 
        errs.append(np.array(act_errs)) 

        if save_path: # save for every simulation
            np.savez(save_path, np.array(regrets), np.array(errs) ) 

    return np.array(regrets), np.array(errs) 


# class ContextualBandit(object):

#     def feed_data(self, contexts, actions, rewards):
#         self._num_contexts = contexts.shape[0] 

#         self.contexts = contexts 
#         self.rewards = rewards 
#         self.actions = actions 

#         self.order = range(self.num_contexts) 

#     def reset(self):
#         self.order = np.random.permutation(self.number_contexts)

#     def context(self, number):
#         return self.contexts[self.order[number], :].reshape(1,-1) 

#     def reward(self, number):
#         return self.rewards[self.order[number], :].reshape(1,-1)

#     def action(self, number):
#         return self.actions[self.order[number], :].reshape(1,-1) 

#     def feed_test_data(self, test_contexts, test_vals):
#         self.test_contexts = test_contexts 
#         self.test_vals = test_vals 

#     @property 
#     def num_contexts(self):
#         return self._num_contexts


class OfflineContextualBandit(object):
    def __init__(self, noise_std, contexts, actions, mean_rewards, test_contexts, test_mean_rewards):
        """
        Args:
            contexts: (None, context_dim) 
            actions: (None,) 
            mean_rewards: (None, num_actions) 
            test_contexts: (None, context_dim)
            test_mean_rewards: (None, num_actions)
        """
        self.noise_std = noise_std
        self.set_data(contexts, actions, mean_rewards, test_contexts, test_mean_rewards)
        self.reset_rewards()
        self.order = range(self.num_contexts) 

    def reset_order(self): 
        self.order = np.random.permutation(self.num_contexts)

    def set_data(self, contexts, actions, mean_rewards, test_contexts, test_mean_rewards): 
        self.set_contexts(contexts) 
        self.set_actions(actions)
        self.set_mean_rewards(mean_rewards)
        self.set_test_contexts(test_contexts)
        self.set_test_mean_rewards(test_mean_rewards) 

    def get_data(self, number): 
        ind = self.order[number]
        a = self.actions[ind]
        return self.contexts[ind:ind+1], self.actions[ind:ind+1], self.rewards[ind:ind+1, a:a+1] 

    def set_contexts(self, contexts): 
        self.contexts = contexts 

    def set_actions(self, actions): 
        self.actions = actions 

    def set_test_contexts(self, test_contexts):
        self.test_contexts = test_contexts

    def set_test_mean_rewards(self, test_mean_rewards): 
        self.test_mean_rewards = test_mean_rewards

    def set_mean_rewards(self, mean_rewards):
        self.mean_rewards = mean_rewards 

    def reset_rewards(self): 
        self.rewards = self.mean_rewards + self.noise_std * np.random.normal(size=self.mean_rewards.shape)

    def set_rewards(self, rewards): 
        self.rewards = rewards

    @property 
    def num_contexts(self): 
        return self.contexts.shape[0] 

    @property 
    def num_actions(self):
        return self.mean_rewards.shape[1] 
    
    @property  
    def context_dim(self):
        return self.contexts.shape[1]

    @property 
    def num_test_contexts(self): 
        return self.test_contexts.shape[0]






