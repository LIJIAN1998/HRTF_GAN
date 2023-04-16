#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=1:men=8gb

module load anaconda3/personal
source activate hrtf_env
python test.py