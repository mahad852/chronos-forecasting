#!/bin/bash
# FILENAME:  fed_avg.slurm

#SBATCH -A cis230083-gpu                # allocation name
#SBATCH --nodes=1                       # Total # of nodes 
#SBATCH --ntasks-per-node=1             # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=2               # Number of GPUs per node
#SBATCH --time=48:00:00                 # Total run time limit (hh:mm:ss)
#SBATCH -J fed_avg_exp                  # Job name
#SBATCH -o slurm_outputs/fed_avg.out    # Name of stdout output file
#SBATCH -e slurm_outputs/fed_avg.err    # Name of stderr error file
#SBATCH -p gpu                          # Queue (partition) name
#SBATCH --mail-user=ma649596@ucf.edu
#SBATCH --mail-type=all                 # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load anaconda
conda activate chronos

# Launch GPU code
python fed_avg_exp.py