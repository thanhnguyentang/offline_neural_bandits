"""Define an abstract class for bandit algorithms."""

class BanditAlgorithm(object):
    def __init__(self):
        pass 

    def sample_action(self, contexts):
        """Choose an action given a context. """
        pass 

    def update_buffer(self, contexts, actions, rewards):
        """Update the internel buffer.""" 
        pass 

    def update(self, contexts, actions, rewards):
        """Update its internal model. """
        pass

    def monitor(self, contexts=None, actions=None, rewards=None):
        """Show some internal values for debugging. """
        pass 

    def reset(self, seed):
        pass 

