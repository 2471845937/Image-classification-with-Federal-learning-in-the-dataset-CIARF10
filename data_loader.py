import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import Subset, DataLoader

def non_iid_split(num_clients=2, alpha=0.5):
    """使用Dirichlet分布划分Non-IID数据"""
    dataset = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    labels = np.array([y for _, y in dataset])
    class_indices = [np.where(labels == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_clients)]

    # 按Dirichlet分布分配样本
    for cls_idx in class_indices:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(cls_idx)).astype(int)
        split_points = np.cumsum(proportions)[:-1]
        np.random.shuffle(cls_idx)
        splits = np.split(cls_idx, split_points)
        for c in range(num_clients):
            if len(splits) > c:
                client_indices[c].extend(splits[c].tolist())

    # 创建客户端数据集
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    return client_datasets

def get_testloader():
    """全局测试集"""
    testset = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())
    return DataLoader(testset, batch_size=32)