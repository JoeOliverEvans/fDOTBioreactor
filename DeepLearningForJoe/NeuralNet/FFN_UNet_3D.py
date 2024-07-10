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


class FFN(nn.Module):
    """ To replicate the classical reconstruction, generates image 1 quarter of the size due to parameter numbers"""
    def __init__(self):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(2592*2, 12*12*14)  # Size of the input /4 in each dimension
        self.upconvo1 = nn.ConvTranspose3d(1, 1, 4, 1)

    def forward(self, x):
        f1 = relu(self.fc1(x))
        f1_reshaped = torch.reshape(f1, [12, 12, 14])
        c1 = relu(self.upconvo1(f1_reshaped))
        return c1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(FFN(), UNet())

    def forward(self, inputx):
        return self.seq(inputx)


class mydata(Dataset):
    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.device = device

    def __len__(self):
        return self.Y.shape[-1]

    def __getitem__(self, idx):
        return torch.unsqueeze(self.X[:, idx], 0).to(self.device), torch.unsqueeze(self.Y[:, idx], 0).to(
            self.device)


def train_loop(dataloader, dataloader_test, model, mask, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        # loss = loss_fn(pred, y)
        loss = torch.zeros(1)
        if torch.cuda.is_available():
            loss = loss.to("cuda")
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                tmp1 = torch.flatten(pred[i, j] * mask)
                tmp2 = torch.flatten(y[i, j] * mask)
                loss = loss.add(torch.sum((tmp1 - tmp2) ** 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if np.isnan(loss):
                break
            print(f"loss: {loss:>7f}  [{current:>4d}/{size:>4d}]")

    test_loss = torch.zeros(1)
    if torch.cuda.is_available():
        test_loss = test_loss.to("cuda")
    for _, (X, y) in enumerate(dataloader_test):
        pred = model(X)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                tmp1 = torch.flatten(pred[i, j] * mask)
                tmp2 = torch.flatten(y[i, j] * mask)
                test_loss = test_loss.add(torch.sum((tmp1 - tmp2) ** 2))

    test_loss = test_loss.item()
    print(f"test loss: {test_loss:>7f}")

    return loss.item(), test_loss


if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # device = ("cpu")
    print(f"Using {device} device")

    # data = sio.loadmat('../SimData/3D/images.mat')
    data_string = r'Datasets/Blob/Blob_10_1/images3_blobs10_joined.mat'
    print(data_string)
    data = mat73.loadmat(data_string)
    training_X = torch.tensor(data['all_datafl_clean'][:, :2048] + data['all_datax_clean'][:, :2048], dtype=torch.float32)
    training_Y = torch.tensor(data['clean_img'][:, :, :, :2048], dtype=torch.float32)
    validation_X = torch.tensor(data['all_datafl_clean'][:, 2048:2400] + data['all_datax_clean'][:, 2048:2400], dtype=torch.float32)
    validation_Y = torch.tensor(data['clean_img'][:, :, :, 2048:2400], dtype=torch.float32)
    # inmesh = np.int16(data['inmesh'].squeeze())
    mask = torch.tensor(data['mask'], dtype=torch.float32).to(device)

    model = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataloader = DataLoader(mydata(training_X, training_Y, device), batch_size=16)
    validate_dataloader = DataLoader(mydata(validation_X, validation_Y, device), batch_size=1)

    all_loss = []
    all_testloss = []
    mintest = np.inf
    for epoch in range(200):
        print(f'Epoch {epoch:>2d}')
        trainloss, testloss = train_loop(train_dataloader, validate_dataloader, model, mask, optimizer)
        all_loss.append(trainloss)
        all_testloss.append(testloss)
        if testloss < mintest:
            mintest = testloss
            patience = 10  # Reset patience counter
        # if epoch>5:
        #     if np.all(np.array(all_testloss[-5:])>mintest):
        #         print('Test loss exceeds minimum for 5 consecutive epochs. Terminating.')
        #         break
        else:
            patience -= 1
            if patience == 0:
                break

    model = model.to('cpu')
    model.eval()

    path_root_string = re.search('.*(?=\/)', data_string).group() + r'/'
    model_path = path_root_string + r'end-to-end/3D_UNet_trained3'
    torch.save(model, model_path)
    sio.savemat(path_root_string + 'loss_3D_UNet3.mat', {'training_loss': all_loss, 'testing_loss': all_testloss})

    # %%
    # Now process the test set
    model = torch.load(model_path)
    test_X = torch.tensor(data['all_datafl_clean'][:, 2400:] + data['all_datax_clean'][:, 2400:], dtype=torch.float32)
    test_Y = np.zeros(test_X.shape)
    for i in range(test_X.shape[-1]):
        tmp = test_X[:, :, :, i]
        test_Y[:, :, :, i] = model(tmp.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()

    sio.savemat(path_root_string + r'end-to-end/test_processed.mat', {'recon2': test_Y})
