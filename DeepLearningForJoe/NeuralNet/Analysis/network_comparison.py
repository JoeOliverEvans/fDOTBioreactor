import torch
import numpy as np
import pandas as pd
from pathlib import Path

"""Plan for this file is to compare the results of different training data on the performance:

must evaluate the testing data for each network and vice versa
the end will be a large comparison table
it will rely on the file structure to do this comparison
consider saving and loading the matrix so that this doesn't have to evaluated every time"""

folder = Path('../Datasets')
dirs_iter = [f for f in folder.iterdir() if f.is_dir()]
try:
    comparison_matrix = pd.read_csv('comparison_matrix.csv')
except FileNotFoundError:
    comparison_matrix = pd.DataFrame()

print(dirs_iter)
print(comparison_matrix)
comparison_matrix.to_csv('comparison_matrix.csv')
