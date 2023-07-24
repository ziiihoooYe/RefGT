#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH --job-name=DRTT

#set number of GPUs
#SBATCH --gres=gpu:8

#mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#send mail to this address
#SBATCH --mail-user=1045406918@qq.com

#set the partition of the job
#SBATCH --partition=big

#run the application
python mail.py
