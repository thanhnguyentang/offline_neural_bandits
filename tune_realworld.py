import os
import subprocess
import time
import argparse
import random
import glob
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='run_exps', choices=['run_exps','collect_results'])
parser.add_argument('--data_types', nargs='+', type=str, default=['mushroom'])
parser.add_argument('--algo_groups', nargs='+', type=str, default=['approx-neural'])
parser.add_argument('--num_sim', type=int, default=3)
parser.add_argument('--models_per_gpu', type=int, default=6)
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='gpus indices used for multi_gpu')

parser.add_argument('--result_dir', type=str, default='results/stock_d=21_a=8_pi=eps-greedy0.1_std=0.1', help='result directory for collect_results()')
args = parser.parse_args()

def multi_gpu_launcher(commands,gpus,models_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    procs = [None]*len(gpus)*models_per_gpu

    while len(commands) > 0:
        for i,proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this index; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs[i] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()


def create_commands(data_type='mushroom', algo_group='approx-neural', num_sim=3):
    commands = []
    if algo_group == 'approx-neural':
        for batch_size,num_steps,buffer_s in [(1,1,1),(50,100,-1)]:
            for beta in [0.01, 0.05, 0.1, 0.5, 1, 5]:
                commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} --beta {}'.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s,beta))

    elif algo_group == 'neural-greedy':
        for batch_size,num_steps,buffer_s in [(1,1,1),(50,100,-1)]:
            commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --batch_size {} --num_steps {} --buffer_s {} '.format(data_type,algo_group,num_sim,batch_size,num_steps,buffer_s))

    elif algo_group == 'kern':
        for rbf_sigma in [0.1, 1,10]:
            commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {} --rbf_sigma {}'.format(data_type,algo_group,num_sim,rbf_sigma))

    elif algo_group == 'baseline': # no tuning 
        commands.append('python realworld_main.py --data_type {} --algo_group {} --num_sim {}'.format(data_type,algo_group,num_sim))

    else:
        raise NotImplementedError

    return commands


def run_exps():
    commands = []
    for data_type in args.data_types:
        for algo_group in args.algo_groups:
            commands += create_commands(data_type, algo_group, args.num_sim)
    random.shuffle(commands)
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)

def collect_results():
    filenames = glob.glob(os.path.join(args.result_dir,"*.npz"))
    results = {}
    for filename in filenames:
        k = np.load(filename)
        regret = k['arr_0'][:,1,:]
        regret = np.min(regret,1) # best regret of a run
        regret = np.mean(regret)
        results[filename] = regret
    
    filenames.sort(key=lambda x: results[x])
    
    for filename in filenames:
        print('{}:   {}'.format(filename,results[filename]))

if __name__ == '__main__':
    eval(args.task)()
