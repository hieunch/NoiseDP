# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

import torch
import numpy as np
import os
from agm import calibrateAnalyticGaussianMechanism
import wandb

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

"""
1. load_data
2. generate clients (step 3)
3. generate aggregator
4. training
"""
client_num = 4
d = load_cnn_mnist(client_num)

"""
FL model parameters.
"""
lr = 0.15
clip = 1
batch_size = len(d[0][0])

### Noise parameters
eps = 0.5
delta = 1e-9
noise_scheme = 'binomial'

### Quatization bits: int or 'adapt'
n_bits = 8

fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': MNIST_CNN,         # model
    'data': d,           # dataset
    'lr': lr,            # learning rate
    'E': 1,              # number of local iterations
    'C': 1,
    'eps': eps,          # privacy budget
    'delta': delta,      # approximate differential privacy: (epsilon, delta)-DP
    'q': 1,              # sampling rate
    'clip': clip,        # clipping norm
    'tot_T': 2000,       # number of aggregation times (communication rounds)
    'batch_size': batch_size,
    'device': device,
    'n_bits': n_bits,
    'noise_scheme': noise_scheme,
}

if noise_scheme == 'gaussian':
    fl_param['sigma'] = calibrateAnalyticGaussianMechanism(eps, delta, lr*clip/batch_size*2)
elif noise_scheme == 'binomial':
    fl_param['trials'] = 3054
    fl_param['success_prob'] = 0.5


run = wandb.init(
    project = "AQDP_Binomial",
    # group = "eps_0.5",
    config = fl_param
)


fl_entity = FLServer(fl_param).to(device)


import time

start_time = time.time()
total_bits = 0
for t in range(fl_param['tot_T']):
    accuracy, precision, recall, f1, loss = fl_entity.global_update()
    msg = {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_loss": loss}
    if n_bits == 'adapt':
        msg["bits_communication"] = fl_entity.n_bits
        print('n_bits =', fl_entity.n_bits)
    total_bits += fl_entity.n_bits
    wandb.log(msg)

    print("global epochs = {:d}, acc = {:.4f}".format(t+1, accuracy), " Time taken: %.2fs" % (time.time() - start_time), " Avg. bits: %.2f" % (total_bits / (t+1)))



# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, fl_param['tot_T'] + 1), acc, marker='o', linestyle='-')
# plt.title('Learning Curve')
# plt.xlabel('Global Epochs')
# plt.ylabel('Accuracy')
# plt.grid()
# # plt.xticks(range(1, fl_param['tot_T'] + 1))  # Optional: to show each epoch on the x-axis
# plt.savefig('acc.png')