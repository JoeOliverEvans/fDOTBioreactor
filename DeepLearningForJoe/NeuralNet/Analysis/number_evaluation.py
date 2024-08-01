from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
import re
import mat73
import scipy.ndimage as ndimage
from matplotlib.patches import Patch



def calculate_num_inclusions(model_output, min_size=1):
    """
    Calculates the number of inclusions in the deblurred output
    :param model_output:
    :param min_size: minimum number of voxels in an inclusion
    :return:
    """
    num_inclusions = []
    for i in range(0, model_output.shape[3]):
        _, labels = clustering(model_output[:, :, :, i], min_size)
        num_inclusions.append(len(labels))
    return num_inclusions


def clustering(matrix, min_number_of_labels):
    """
    Returns the labeled array and a list of labels which have over the minimum number of labels
    :param min_number_of_labels:
    :param matrix:
    :return:
    """
    connection_matrix = [[[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]],
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]],
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]]
    labelled_matrix = ndimage.label(matrix, connection_matrix)
    label_list = significant_labels(labelled_matrix[0], labelled_matrix[1], min_number_of_labels)
    return np.array(labelled_matrix[0]), label_list


def significant_labels(matrix, number_of_labels, min_number_of_labels):
    relevant_labels = []
    for x in range(number_of_labels + 1):
        if np.count_nonzero(matrix == x) >= min_number_of_labels and x != 0:
            relevant_labels.append(x)
    return relevant_labels


def calculate_evaluation(clean_images, model_output):
    """
    Evaluates with MSE loss
    :param clean_images:
    :param model_output:
    :return: evaluation_results
    """
    evaluation_results = []
    for i in range(np.shape(clean_images)[3]):
        evaluation_results.append(np.sum((clean_images[:, :, :, i] - model_output[:, :, :, i]) ** 2))
    return evaluation_results


if __name__ == '__main__':
    file_path = r'../../SimulateData/Experiments/a_max3_vs_only3/'

    ground_truth = mat73.loadmat(
        file_path + r'images3_blobs_testing.mat')  # Array containing the ground truth number of inclusions
    ground_truth_all_nblobs = ground_truth['all_nblob']
    ground_truth_clean_img = ground_truth['clean_img']

    output_model1 = sio.loadmat(file_path + r'test_processed_max3_testing.mat')[
        'recon2']  # Array containing the reconstruction
    output_model2 = sio.loadmat(file_path + r'test_processed_only3_testing.mat')['recon2']

    num_inclusions_model1 = calculate_num_inclusions(output_model1 > 2, 20)
    num_inclusions_model2 = calculate_num_inclusions(output_model2 > 2, 20)
    print(num_inclusions_model1)
    evaluation_model1 = calculate_evaluation(ground_truth_clean_img, output_model1)
    evaluation_model2 = calculate_evaluation(ground_truth_clean_img, output_model2)

    fig, ax1 = plt.subplots()

    for i in range(10):
        bplot_tmp_1 = ax1.boxplot(num_inclusions_model1[i * 20:i * 20 + 20], positions=[i + 1.25], patch_artist=True)
        bplot_tmp_2 = ax1.boxplot(num_inclusions_model2[i * 20:i * 20 + 20], positions=[i + 0.75], patch_artist=True)
        bplot_tmp_1['boxes'][0].set_facecolor('red')
        bplot_tmp_2['boxes'][0].set_facecolor('blue')
    ax1.plot([0.75, 9.75], [1, 10])
    ax1.plot([1.25, 10.25], [1, 10], color='orange')
    ax1.set_title('Number of inclusions reconstructed vs ground truth number')
    ax1.set_xlabel('Ground truth number of inclusions')
    ax1.set_ylabel('Reconstructed number of inclusions (boxplot)')
    ax1.set_xticklabels([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
    ax1.set_ylim(0, 10)

    fig.tight_layout()
    plt.legend(handles=[Patch(facecolor='blue', edgecolor='none', label='Max3'),
                Patch(facecolor='red', edgecolor='none', label='Only3')])
    plt.savefig(file_path + r'InclusionNumberComparison.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.show()

    avg_model1 = []
    std_model1 = []
    avg_model2 = []
    std_model2 = []
    for i in range(10):
        avg_model1.append(np.mean(evaluation_model1[i * 20:i * 20 + 20]))
        std_model1.append(np.std(evaluation_model1[i * 20:i * 20 + 20]))
        avg_model2.append(np.mean(evaluation_model2[i * 20:i * 20 + 20]))
        std_model2.append(np.std(evaluation_model2[i * 20:i * 20 + 20]))

    plt.errorbar(np.arange(10) + 1, avg_model1, yerr=std_model1, color='blue', label='max3', fmt='x', ecolor='k',
                 capsize=5)
    plt.errorbar(np.arange(10) + 1, avg_model2, yerr=std_model2, color='red', label='only3', fmt='x', ecolor='k',
                 capsize=5)
    plt.xlabel('Ground truth number of inclusions')
    plt.ylabel('MSE reconstruction loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
