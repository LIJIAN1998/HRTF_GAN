import torch
import subprocess 
# print("using cuda? ", torch.cuda.is_available())

# Define the MATLAB command
matlab_cmd = "/rds/general/user/jl2622/projects/sonicom/live/matlab/R2021a/bin/matlab -nosplash -nodisplay -nojvm -nodesktop -r"

# Define the MATLAB script to be executed
matlab_script = """
addpath('./evaluation/test.m');
[num, str] = deal({0}, '{1}');
[result_int, result_str] = multiply_and_concat(num, str);
disp(result_int);
disp(result_str);
exit;
"""

# Define the input values
input_int = 5
input_str = "example"

matlab_script_path = "./evaluation/test.m"
matlab_script = ""

# Format the MATLAB script command with the input values
matlab_cmd = matlab_cmd + ' "{0}"'.format(matlab_script.format(input_int, input_str))

# Call the MATLAB script using subprocess
try:
    output = subprocess.check_output(matlab_cmd, shell=True, stderr=subprocess.STDOUT)
    output = output.decode("utf-8")
    print("MATLAB output:", output)
    # result_int, result_str = output.strip().split("\n")[-2:]
    # print("Result (integer):", int(result_int))
    # print("Result (string):", result_str)
except subprocess.CalledProcessError as e:
    print("Error:", e.output)
