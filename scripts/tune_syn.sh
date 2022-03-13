#!/bin/bash
#SBATCH --job-name="neuralin-syn"
#SBATCH --output=gpu_job.out
#SBATCH --error=gpu_job.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --export=ALL

cd ~/deep_bandits
# module load ~/jaxv/bin/python
source ~/jaxv/bin/activate
export PYTHONHOME=~/jaxv
export PYTHONPATH="~/jaxv/bin/python"
echo $PYTHONHOME
echo $PYTHONPATH
which python
python tune_synthetic.py --data_types quadratic quadratic2 cosine --algo_groups neurallinlcb 
