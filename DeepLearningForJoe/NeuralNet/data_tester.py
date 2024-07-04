import mat73
import torch


data_string = r'C:\Joe Evans\University\Computing\Summer Project\DeepLearningForJoe\NeuralNet\Datasets\Gaussian\Gaussian_20_1\images3_gaussian2500new.mat'
data = mat73.loadmat(data_string)
training_X = torch.tensor(data['noisy_img'][:, :, :, :5], dtype=torch.float32)
training_Y = torch.tensor(data['clean_img'][:, :, :, :5], dtype=torch.float32)
validation_X = torch.tensor(data['noisy_img'][:, :, :, 5:200], dtype=torch.float32)
validation_Y = torch.tensor(data['clean_img'][:, :, :, 5:200], dtype=torch.float32)

print(training_X.min())
print(training_Y.min())
print(validation_X.min())
print(validation_Y.min())
