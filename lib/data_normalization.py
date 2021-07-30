import numpy as np
import torch
from torchvision import transforms


def get_transform(mean, std):
    if isinstance(mean, np.ndarray):
        mean = mean.tolist()
    if isinstance(mean, torch.Tensor):
        mean = mean.tolist()
    elif type(mean) is not list:
        mean = [mean]

    if isinstance(std, np.ndarray):
        std = std.tolist()
    if isinstance(std, torch.Tensor):
        std = std.tolist()
    elif type(std) is not list:
        std = [std]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return data_transform


def denormalize_torch(data, mean, std):
    if isinstance(std, torch.Tensor) or isinstance(std, list):
        data_denorm = data.clone()

        for i, (mean_i, std_i) in enumerate(zip(mean.tolist(), std.tolist())):
            data_denorm[i, :, :, :] = (data[i, :, :, :] * std_i) + mean_i
    else:
        data_denorm = (data * std) + mean

    return data_denorm


def denormalize_numpy(data, mean, std):
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()

    if isinstance(std, np.ndarray) or isinstance(std, torch.Tensor) or isinstance(std, list):
        data_denorm = np.zeros_like(data)

        for i, (mean_i, std_i) in enumerate(zip(mean.tolist(), std.tolist())):
            data_denorm[i, :, :, :] = (data[i, :, :, :] * std_i) + mean_i
    else:
        data_denorm = (data * std) + mean

    return data_denorm
