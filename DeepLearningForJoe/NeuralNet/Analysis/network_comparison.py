import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import mat73
import scipy.io as sio

"""Plan for this file is to compare the results of different training data on the performance:

must evaluate the testing data for each network and vice versa
the end will be a large comparison table
it will rely on the file structure to do this comparison
consider saving and loading the matrix so that this doesn't have to evaluated every time"""


def get_test_file_name(folder):
    """
    Returns the name of the test data file and generates one if it doesn't exist
    :param folder:
    :return:
    """
    files_in_folder = os.listdir(folder)
    name_of_test = ''
    test_file_present = False
    for file_name in files_in_folder:
        if file_name[-9:-4] != '_test':
            test_file_present = True
            name_of_test = file_name
            break
    if test_file_present:
        return name_of_test
    else:
        return get_test_file_name(files_in_folder)


def generate_test_file(file_list):
    """
    Makes the test file separate to the data file for faster loading times
    :param file_list: list of file names
    :return: test file name
    """
    print(f'No test file present in {dataset_folder}, generating test file')
    test_file_name = ''
    for file_name in file_list:
        if file_name[0:7] == 'images3_':
            data = mat73.loadmat(file_name)
            noisy = data['noisy_img'][:, :, :, 2400:]
            clean = data['clean_img'][:, :, :, 2400:]
            test_file_name = str(dataset_folder) + file_name[:-4] + r'_test.mat'
            sio.savemat(test_file_name, {'noisy_img': noisy, 'clean_img': clean})
            del data, noisy, clean
            break
    return test_file_name


try:
    comparison_matrix = pd.read_csv('comparison_matrix.csv')
except FileNotFoundError:
    comparison_matrix = pd.DataFrame()

folder_ = Path('../Datasets')

model_type_path_list = [f for f in folder_.iterdir() if (f.is_dir() and str(f)[0] != '_')]

for model_type_path in model_type_path_list:
    model_folder_list = [f for f in model_type_path.iterdir() if (f.is_dir() and str(f)[0] != '_')]
    for model_folder in model_folder_list:
        model_path = str(model_folder) + '3D_UNet_trained3'
        model = torch.load(model_path)

        '''
        dataset_type_path_list = [f for f in folder.iterdir() if (f.is_dir() and str(f)[0] != '_')]     # _ is the ignore character
        
        for dataset_type_path in dataset_type_path_list:
            dataset_folder_list = [f for f in dataset_type_path.iterdir() if (f.is_dir() and str(f)[0] != '_')]
            for dataset_folder in dataset_folder_list:
                test_name = get_test_file_name(dataset_folder)
                test_data = sio.loadmat(test_name)
        
                test_X = torch.tensor(test_data['noisy_img'], dtype=torch.float32)
                test_Y = np.zeros(test_X.shape)
                for i in range(test_X.shape[-1]):
                    tmp = test_X[:, :, :, i]
                    test_Y[:, :, :, i] = model(tmp.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()
        
                cumulative_error = 0
                for i in range(0, 100):
                    cumulative_error += np.sum(np.abs(test_X[:, :, :, i] - test_Y[:, :, :, i]))     # RMSE




# Create the test file, will make loading that dataset faster for all the models that need to use it

print(comparison_matrix)
comparison_matrix.to_csv('comparison_matrix.csv')'''
