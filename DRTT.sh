#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH -J DRTT

# set number of GPUs
#SBATCH --gres=gpu:8

# set job partition
#SBATCH --partition=big

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=ziiihooo.ye@gmail.com

source /jmain02/home/J2AD007/txk47/zxy40-txk47/miniconda3/etc/profile.d/conda.sh

#activae conda env
conda activate DRTT

# run the main code
# torchrun --standalone --nproc_per_node=8 main.py --dataset=SPA-Data
        #  --test=True --save_dir=test_result --model_path=save_temp/msssim0.5psnr0.3/model/model_00075.pt

# cd baseline/model/PReNet
# train baseline
python baseline/model/PReNet/train_PReNet.py --dataset=SPA-Data
