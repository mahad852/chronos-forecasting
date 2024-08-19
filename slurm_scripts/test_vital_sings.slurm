#!/bin/bash
# FILENAME:  fed_avg.slurm

#SBATCH -A cis230083-gpu                # allocation name
#SBATCH --nodes=1                       # Total # of nodes 
#SBATCH --ntasks-per-node=1             # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=2               # Number of GPUs per node
#SBATCH --time=30:00:00                 # Total run time limit (hh:mm:ss)
#SBATCH -J fed_avg_exp                  # Job name
#SBATCH -o slurm_outputs/test_vital_signs.out    # Name of stdout output file
#SBATCH -e slurm_outputs/test_vital_signs.err    # Name of stderr error file
#SBATCH -p gpu                          # Queue (partition) name
#SBATCH --mail-user=ma649596@ucf.edu
#SBATCH --mail-type=all                 # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load anaconda/2021.05-py38
conda activate chronos

# Launch GPU code
python test_vital_signs.py