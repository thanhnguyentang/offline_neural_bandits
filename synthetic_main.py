"""Mini main for testing algorithms. """ 

import numpy as np 
import jax  
import jax.numpy as jnp 
from easydict import EasyDict as edict
import os 

from core.contextual_bandit import contextual_bandit_runner
from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2
from algorithms.lin_lcb import LinLCB 
from algorithms.kern_lcb import KernLCB 
from algorithms.uniform_sampling import UniformSampling
from algorithms.neural_lin_lcb import ExactNeuralLinLCBV2, ExactNeuralLinGreedyV2, ApproxNeuralLinLCBV2, ApproxNeuralLinGreedyV2, \
    ApproxNeuralLinLCBJointModel, NeuralLinGreedyJointModel
from data.synthetic_data import SyntheticData

from absl import flags, app
import os 
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
# try:
#     jnp.linalg.qr(jnp.array([[0, 1], [1, 1]]))
# except RuntimeError:
#     pass 

FLAGS = flags.FLAGS 

flags.DEFINE_string('data_type', 'quadratic', 'Dataset to sample from')
flags.DEFINE_string('policy', 'eps-greedy', 'Offline policy, eps-greedy/subset')
flags.DEFINE_float('eps', 0.1, 'Probability of selecting a random action in eps-greedy')
flags.DEFINE_float('subset_r', 0.5, 'The ratio of the action spaces to be selected in offline data')
flags.DEFINE_integer('num_contexts', 10000, 'Number of contexts for training.') 
flags.DEFINE_integer('num_test_contexts', 10000, 'Number of contexts for test.') 
flags.DEFINE_boolean('verbose', True, 'verbose') 
flags.DEFINE_boolean('debug', True, 'debug') 
flags.DEFINE_boolean('normalize', False, 'normalize the regret') 
flags.DEFINE_integer('update_freq', 1, 'Update frequency')
flags.DEFINE_integer('freq_summary', 10, 'Summary frequency')

flags.DEFINE_integer('test_freq', 10, 'Test frequency')
flags.DEFINE_string('algo_group', 'approx-neural', 'baseline/neural')
flags.DEFINE_integer('num_sim', 10, 'Number of simulations')
flags.DEFINE_float('noise_std', 0.1, 'Noise std')

flags.DEFINE_integer('chunk_size', 500, 'Chunk size')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_steps', 100, 'Number of steps to train NN.') 
flags.DEFINE_integer('buffer_s', -1, 'Size in the train data buffer.')
flags.DEFINE_bool('data_rand', True, 'Where randomly sample a data batch or  use the latest samples in the buffer' )

flags.DEFINE_float('rbf_sigma', 1, 'RBF sigma for KernLCB') # [0.1, 1, 10]

# NeuraLCB 
flags.DEFINE_float('beta', 0.1, 'confidence paramter') # [0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
flags.DEFINE_float('lr', 1e-3, 'learning rate') 
flags.DEFINE_float('lambd0', 0.1, 'minimum eigenvalue') 
flags.DEFINE_float('lambd', 1e-4, 'regularization parameter')

#================================================================
# Network parameters
#================================================================
def main(unused_argv): 

    #=================
    # Data 
    #=================
    if FLAGS.policy == 'eps-greedy':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.eps)
    elif FLAGS.policy == 'subset':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.subset_ratio)
    elif FLAGS.policy == 'online':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.eps) 
    else:
        raise NotImplementedError('{} not implemented'.format(FLAGS.policy))

    data = SyntheticData(num_contexts=FLAGS.num_contexts,
                    num_test_contexts=FLAGS.num_test_contexts, 
                    pi = FLAGS.policy, 
                    eps = FLAGS.eps,
                    subset_r = FLAGS.subset_r,
                    noise_std = FLAGS.noise_std, 
                    name = FLAGS.data_type, 
    )
    

    context_dim = data.context_dim 
    num_actions = data.num_actions 
    
    hparams = edict({
        'layer_sizes': [20,20], 
        's_init': 1, 
        'activation': jax.nn.relu, 
        'layer_n': True,
        'seed': 0,
        'context_dim': context_dim, 
        'num_actions': num_actions, 
        'beta': FLAGS.beta, # [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        'lambd': FLAGS.lambd, # regularization param: [0.1m, m, 10 m  ]
        'lr': FLAGS.lr, 
        'lambd0': FLAGS.lambd0, # shoud be lambd/m in theory but we fix this at 0.1 for simplicity and mainly focus on tuning beta 
        'verbose': False, 
        'batch_size': FLAGS.batch_size,
        'freq_summary': FLAGS.freq_summary, 
        'chunk_size': FLAGS.chunk_size, 
        'num_steps': FLAGS.num_steps, 
        'buffer_s': FLAGS.buffer_s, 
        'data_rand': FLAGS.data_rand,
        'debug_mode': 'full' # simple/full
    })

    lin_hparams = edict(
        {
            'context_dim': hparams.context_dim, 
            'num_actions': hparams.num_actions, 
            'lambd0': hparams.lambd0, 
            'beta': hparams.beta,  
            'rbf_sigma': FLAGS.rbf_sigma, # 0.1, 1, 10
            'max_num_sample': 1000 
        }
    )

    data_prefix = '{}_d={}_a={}_pi={}_std={}'.format(FLAGS.data_type, \
            context_dim, num_actions, policy_prefix, data.noise_std)

    res_dir = os.path.join('results', data_prefix) 

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    #================================================================
    # Algorithms 
    #================================================================

    if FLAGS.algo_group == 'approx-neural':
        algos = [
                UniformSampling(lin_hparams),
                # NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq), 
                ApproxNeuraLCBV2(hparams, update_freq = FLAGS.update_freq)
            ]

        algo_prefix = 'approx-neural-gridsearch_epochs={}_m={}_layern={}_buffer={}_bs={}_lr={}_beta={}_lambda={}_lambda0={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, hparams.buffer_s, hparams.batch_size, hparams.lr, \
            hparams.beta, hparams.lambd, hparams.lambd0
        )

    
    if FLAGS.algo_group == 'neural-greedy':
        algos = [
                UniformSampling(lin_hparams),
                NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq), 
            ]

        algo_prefix = 'neural-greedy-gridsearch_epochs={}_m={}_layern={}_buffer={}_bs={}_lr={}_lambda={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, hparams.buffer_s, hparams.batch_size, hparams.lr, \
           hparams.lambd
        ) 


    if FLAGS.algo_group == 'baseline':
        algos = [
            UniformSampling(lin_hparams),
            LinLCB(lin_hparams),
            ## KernLCB(lin_hparams), 
            # NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq),
            # ApproxNeuralLinLCBV2(hparams), 
            # ApproxNeuralLinGreedyV2(hparams),
            NeuralLinGreedyJointModel(hparams), 
            ApproxNeuralLinLCBJointModel(hparams)
        ]

        algo_prefix = 'baseline_epochs={}_m={}_layern={}_beta={}_lambda0={}_rbf-sigma={}_maxnum={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, \
            hparams.beta, hparams.lambd0, lin_hparams.rbf_sigma, lin_hparams.max_num_sample
        )

    if FLAGS.algo_group == 'kern': # for tuning KernLCB
        algos = [
            UniformSampling(lin_hparams),
            KernLCB(lin_hparams), 
        ]

        algo_prefix = 'kern-gridsearch_beta={}_rbf-sigma={}_maxnum={}'.format(
            hparams.beta, lin_hparams.rbf_sigma, lin_hparams.max_num_sample
        )

    if FLAGS.algo_group == 'neurallinlcb': # Tune NeuralLinLCB seperately  
        algos = [
            UniformSampling(lin_hparams),
            ApproxNeuralLinLCBJointModel(hparams)
        ]

        algo_prefix = 'neurallinlcb-gridsearch_m={}_layern={}_beta={}_lambda0={}'.format(
            min(hparams.layer_sizes), hparams.layer_n, hparams.beta, hparams.lambd0
        )


 
    #==============================
    # Runner 
    #==============================
    file_name = os.path.join(res_dir, algo_prefix) + '.npz' 

    regrets, errs = contextual_bandit_runner(algos, data, FLAGS.num_sim, 
        FLAGS.update_freq, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, file_name)

    np.savez(file_name, regrets, errs)


if __name__ == '__main__': 
    app.run(main)
