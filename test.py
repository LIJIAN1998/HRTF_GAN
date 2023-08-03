import argparse
import torch
import json
print("test")
print("using cuda? ", torch.cuda.is_available())

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