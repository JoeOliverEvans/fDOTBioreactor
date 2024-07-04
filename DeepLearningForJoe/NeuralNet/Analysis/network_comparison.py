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


def generate_test_file(file_list):
    """
    Makes the test file separate to the data file for faster loading times
    :param file_list: list of file names
    :return:
    """
    print(f'No test file present in {dirs_iter[i]}, generating test file')
    for file_name in file_list:
        if file_name[0:7] == 'images3_':
            data = mat73.loadmat(file_name)
            noisy = data['noisy_img'][:, :, :, 2400:]
            clean = data['clean_img'][:, :, :, 2400:]
            sio.savemat(str(dirs_iter[i]) + file_name[:-4] + r'_test.mat', {'noisy_img': noisy, 'clean_img': clean})
            del data, noisy, clean
            break
    return


folder = Path('../Datasets')
dirs_iter = [f for f in folder.iterdir() if f.is_dir()]
try:
    comparison_matrix = pd.read_csv('comparison_matrix.csv')
except FileNotFoundError:
    comparison_matrix = pd.DataFrame()

i = 0

files_in_folder = os.listdir(dirs_iter[i])
test_file_present = False
for file_name in files_in_folder:
    if file_name[-9:-4] != '_test':
        test_file_present = True
        break

# Create the test file, will make loading that dataset faster for all the models that need to use it
if not test_file_present:
    generate_test_file(files_in_folder)



print(dirs_iter)
print(comparison_matrix)
comparison_matrix.to_csv('comparison_matrix.csv')
