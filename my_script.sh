#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jl2622 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jl2622/test_env/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
# python main.py debug --hpc False --tag ari-upscale-4
python test.py