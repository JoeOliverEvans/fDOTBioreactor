from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import re
import mat73
import scipy.ndimage as ndimage


def calculate_num_inclusions(model_output):
    """
    Calculates the number of inclusions in the deblurred output
    :param model_output:
    :return:
    """
    num_inclusions = []
    for image in model_output:
        num_inclusions.append(
            np.max(ndimage.label(image)))  # TODO perhaps implement a threshold to the size of a sphere
    return num_inclusions


def calculate_evaluation(clean_images, model_output):
    """
    Evaluates with MSE loss
    :param clean_images:
    :param model_output:
    :return: evaluation_results
    """
    evaluation_results = []
    for truth, output in zip(clean_images, model_output):
        evaluation_results.append(np.sum((output-truth)**2))
    return evaluation_results


num = 2400

file_path = r'C:\Joe Evans\error'

ground_truth = mat73.loadmat(file_path + r'.mat') # Array containing the ground truth number of inclusions
ground_truth_all_nblobs = ground_truth['all_nblob'][num:]
ground_truth_clean_img = ground_truth['clean_img'][num:]

output_model1 = mat73.loadmat(file_path + r'test_processed.mat')['recon2'] # Array containing the reconstruction
output_model2 = mat73.loadmat(file_path + r'test_processed.mat')['recon2']

num_inclusions_model1 = calculate_num_inclusions(output_model1)
num_inclusions_model2 = calculate_num_inclusions(output_model2)

evaluation_model1 = calculate_evaluation(ground_truth_clean_img, output_model1)
evaluation_model2 = calculate_evaluation(ground_truth_clean_img, output_model2)

fig, ax1 = plt.subplots()

ax1.boxplot((ground_truth_all_nblobs, num_inclusions_model1), color='blue', label='model1')
ax1.boxplot((ground_truth_all_nblobs, num_inclusions_model2), color='red', label='model2')
fig.title('Comparison of number of inclusions reconstructed vs ground truth number')
ax1.set_xlabel('Ground truth number of inclusions')
ax1.set_ylabel('Reconstructed number of inclusions (boxplot)')

ax2 = ax1.twinx()
ax2.violinplot((ground_truth_all_nblobs, evaluation_model1), color='blue', label='model1')
ax2.violinplot((ground_truth_all_nblobs, evaluation_model2), color='red', label='model2')
ax2.set_ylabel('MSE loss (violin plot)')

fig.tight_layout()
plt.legend()
plt.savefig(file_path + r'InclusionNumberComparison.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()

