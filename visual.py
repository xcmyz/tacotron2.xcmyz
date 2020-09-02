import os
import torch
import argparse
import matplotlib
import random

import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, step, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')

    os.makedirs("image", exist_ok=True)
    plt.savefig(os.path.join("image", str(step)+".jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()
    file_list = os.listdir("attention")
    data_file_list = list()
    for filename in file_list:
        if str(args.step) == filename[:len(str(args.step))]:
            data_file_list.append("attention/"+filename)
    data_list = list()
    for filename in data_file_list:
        data_list.append(np.load(filename))
    data_list = random.sample(data_list, 3)
    plot_data(data_list, args.step)

    # plot_data([torch.eye(123, 234) for _ in range(2)], 0)
