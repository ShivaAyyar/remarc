#!/bin/bash
#SBATCH -t 0-24:00 #
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH --job-name random_landscape # Job name
#SBATCH -o /mnt/vstor/SOM_CCCC_JGS25/sayyar/remarc/log/slurm/%x_%A_%a.out # File to which standard out will be written (dir must exist before sbatch)
#SBATCH -e /mnt/vstor/SOM_CCCC_JGS25/sayyar/remarc/log/slurm/%x_%A_%a.err # File to which standard err will be written
#SBATCH --mail-user=ssa163@case.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --array=0-29%8

LS=(3 20)
NS=(100 10 1 0.1 0.01)
SEEDS=(0 1 2)

nS=${#SEEDS[@]}; nN=${#NS[@]}

L_I=$((SLURM_ARRAY_TASK_ID / (nS * nN)))
NS_I=$(((SLURM_ARRAY_TASK_ID / nS) % nN))
SEED_I=$((SLURM_ARRAY_TASK_ID % nS))

SKIP_RL=""
if [ "${LS[$L_I]}" -eq 20 ]; then SKIP_RL="--skip-rl"; fi

cd /mnt/vstor/SOM_CCCC_JGS25/sayyar/remarc/

source .venv/bin/activate

python examples/new_runtime_comparison_randgen.py --N 2 --L ${LS[$L_I]} --Ns ${NS[$NS_I]} --seeds ${SEEDS[$SEED_I]} $SKIP_RL
