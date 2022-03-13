"""Define an online contextual bandit to collect data."""

import numpy as np
from tqdm import tqdm

def contextual_bandit_runner(algo, num_sim, num_contexts, update_freq, test_freq, verbose, debug, normalize, save_path=None):
    """Run an offline contextual bandit problem on a set of algorithms in the same dataset. 

    Args:
        algo: forced to be LinUCB
        num_sim: Number of simulations 
        num_contexts: Number of samples to collect
    """

    # Create a bandit instance 
    for sim in range(num_sim):
        # Run the offline contextual bandit in an online manner 
        print('Simulation: {}/{}'.format(sim + 1, num_sim))
 
        algo.reset(sim * 1111)


        for i in tqdm(range(num_contexts),ncols=75):
            c = data.sample_context()
            a = algo.sample_action(c)
            r = data.sample_reward(c,a) 

            algo.update(c,a,r)




        if save_path: # save for every simulation
            np.savez(save_path, np.array(regrets), np.array(errs) ) 

    return np.array(regrets), np.array(errs) 