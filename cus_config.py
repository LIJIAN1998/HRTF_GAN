from config import Config
import itertools
import json


def main():
    n = 0
    batch_size = [4, 8, 16]
    optimizer = ["adam", "rmsprop"]
    lr = [1e-4, 1e-3, 1e-2]
    alpha = [1e-2, 1]
    lambda_feature = [1e-3, 1e-2]
    latent_dim = [10, 50, 100]
    critic_iter = [3, 4, 5]

    combinations = list(itertools.product(batch_size, optimizer))

    tag = 'ari-upscale-4'
    config = Config(tag, using_hpc=True)
    for combination in combinations:
        n += 1
        print("index: ", n)
        config.batch_size = combination[0]
        config.optimizer = combination[1]
        config.save(n)
        print(config.batch_size)
        print(config.optimizer)

    for i in range(1, 7):
        config = Config(tag, using_hpc=True)
        print(i)
        config.load(i)
        print(config.batch_size, config.optimizer)
if __name__ == '__main__':
    main()