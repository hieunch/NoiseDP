"""
Useful tools
"""
import numpy as np
import random
import torch
from torchvision import datasets, transforms

def load_cnn_mnist(num_users):
    train = datasets.MNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    train_data = train.data.float().unsqueeze(1)
    train_label = train.targets

    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std

    test = datasets.MNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    test_data = test.data.float().unsqueeze(1)
    test_label = test.targets
    test_data = (test_data - mean) / std

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(train_label, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = train_data[idx]
        targets = train_label[idx].float()
        non_iid.append((d, targets))
        print(d.shape)
    non_iid.append((test_data.float(), test_label.float()))
    return non_iid

def mnist_noniid(labels, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users*3)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def gaussian_noise(data_shape, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma, data_shape).to(device)

def binomial_noise(data_shape, m, p, device=None):
    """
    Gaussian noise
    """
    dist = torch.distributions.binomial.Binomial(m, p)
    return (dist.sample(data_shape) - m*p).to(device)