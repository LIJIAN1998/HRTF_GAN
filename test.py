import argparse
import torch
import json
import torch.nn as nn
print("test")
print("using cuda? ", torch.cuda.is_available())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
x = torch.randn(1, 256, 2116).to(device)
print(x.shape)
conv1 = nn.Sequential(
    nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm1d(256),
    nn.ReLU(),
).to(device)
x = conv1(x)
print(x.shape)
print('done')

# def main(index):
#     with open(f"config_files/config_{index}.json", 'r') as f:
#         j = json.load(f)
#     with open("json_log.txt", 'a') as f:
#         f.write(f"{j}\n")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--index")
#     args = parser.parse_args()
#     main(args.index)