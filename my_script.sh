#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jl2622 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/jl2622/hrtf_env/bin/:$PATH
source activate
source /vol/cuda/11.7.1/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
nvcc -V
uptime
python main_2.py train --hpc False --tag ari-upscale-4
# python test.py
# python model/model.py