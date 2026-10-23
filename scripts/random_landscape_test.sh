#!/bin/bash
#SBATCH -t 0-24:00 #
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --job-name random_landscape # Job name
#SBATCH -o /mnt/vstor/SOM_CCCC_JGS25/sayyar/remarc/log/slurm/%x_%j.out # File to which standard out will be written (dir must exist before sbatch)
#SBATCH -e /mnt/vstor/SOM_CCCC_JGS25/sayyar/remarc/log/slurm/%x_%j.err # File to which standard err will be written
#SBATCH --mail-user=ssa163@case.edu
#SBATCH --mail-type=BEGIN,FAIL,END

cd /mnt/vstor/SOM_CCCC_JGS25/sayyar/remarc/

source .venv/bin/activate

python examples/new_runtime_comparison_randgen.py --N 2 3 --seeds 0 1 2 3 4