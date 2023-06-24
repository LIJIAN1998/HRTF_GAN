import torch
import subprocess 
# print("using cuda? ", torch.cuda.is_available())

matlab_path = '/rds/general/user/jl2622/projects/sonicom/live/matlab/R2021a/bin/matlab'

matlab_script_path = './evaluation/test.m'
command = [matlab_path, '-batch', f"run('{matlab_script_path}')"]
subprocess.run(command)