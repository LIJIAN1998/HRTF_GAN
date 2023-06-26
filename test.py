import torch
import subprocess 
# print("using cuda? ", torch.cuda.is_available())

matlab_path = '/rds/general/user/jl2622/projects/sonicom/live/matlab/R2021a/bin/matlab'

matlab_script_path = './evaluation/test.m'
# command = [matlab_path, '-batch', f"run('{matlab_script_path}')"]
# subprocess.run(command)

param1 = 10
param2 = 'hello'
matlab_command = '/rds/general/user/jl2622/projects/sonicom/live/matlab/R2021a/bin/matlab -nodesktop -nosplash -r'

command = f"{matlab_command} \"try, {matlab_script_path}({param1}, '{param2}'), catch ME, fprintf('%s', ME.message), end, exit\""

result = subprocess.run(command, capture_output=True, text=True)
output = result.stdout.strip()
print(output)
