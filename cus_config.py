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

    combinations = list(itertools.product(batch_size, optimizer, lr, alpha, lambda_feature, latent_dim, critic_iter))

    tag = 'ari-upscale-4'
    config = Config(tag, using_hpc=True)
    for combination in combinations:
        with open("cust_log.txt", "a") as f:
            f.write(f"{n}\n")
        print("index: ", n)
        config.batch_size = combination[0]
        config.optimizer = combination[1]
        config.lr = combination[2]
        config.alpha = combination[3]
        config.lambda_feature = combination[4]
        config.latent_dim = combination[5]
        config.critic_iters = combination[6]
        config.save(n)

if __name__ == '__main__':
    main()