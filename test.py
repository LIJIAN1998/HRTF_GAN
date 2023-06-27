import torch
import subprocess 
# print("using cuda? ", torch.cuda.is_available())

matlab_path = '/rds/general/user/jl2622/projects/sonicom/live/matlab/R2021a/bin/matlab'

matlab_script_path = './evaluation/test.m'

# command = [matlab_path, '-batch', f"run('{matlab_script_path}')"]
# subprocess.run(command)

arg1 = 10
arg2 = 'hello'
command = f"{matlab_script_path}({arg1}, {arg2})"
# command = f"run('{matlab_script_path}', {', '.join(map(repr, parameters))});"
process = subprocess.Popen([matlab_path, '-nodesktop', '-nosplash', '-r', command],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)

# for param in parameters:
#     process.stdin.write(f"{param}\n".encode())
# process.stdin.close()
output = process.stdout.read().decode()

process.wait()
print(output)
# param1 = 10
# param2 = 'hello'
# matlab_command = '/rds/general/user/jl2622/projects/sonicom/live/matlab/R2021a/bin/matlab -nodesktop -nosplash -r'

# command = f"{matlab_command} \"try, {matlab_script_path}({param1}, '{param2}'), catch ME, fprintf('%s', ME.message), end, exit\""

# result = subprocess.run(command, capture_output=True, text=True)
# output = result.stdout.strip()
# print(output)
