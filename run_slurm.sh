#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE # required to send email notifcations
#SBATCH --mail-user= # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/wz1620/myenv/bin/:$PATH
source activate
source /vol/cuda/11.7.1/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
python                  
