import numpy as np
import scipy.io as sio
import mat73
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import re
from UNet_3D import UNet


model_path = r'../SimulateData/Experiments/a_max3_vs_only3/'
data = mat73.loadmat(model_path + r'images3_blobs_testing.mat')

model = torch.load(model_path + r'3D_UNet_trained3_max3')
test_X = torch.tensor(data['noisy_img'], dtype=torch.float32)
test_Y = np.zeros(test_X.shape)
for i in range(test_X.shape[-1]):
    tmp = test_X[:, :, :, i]
    test_Y[:, :, :, i] = model(tmp.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()
sio.savemat(model_path + r'test_processed_max3_testing.mat', {'recon2': test_Y})

model = torch.load(model_path + r'3D_UNet_trained3_only3')
test_X = torch.tensor(data['noisy_img'], dtype=torch.float32)
test_Y = np.zeros(test_X.shape)
for i in range(test_X.shape[-1]):
    tmp = test_X[:, :, :, i]
    test_Y[:, :, :, i] = model(tmp.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()
sio.savemat(model_path + r'test_processed_only3_testing.mat', {'recon2': test_Y})
