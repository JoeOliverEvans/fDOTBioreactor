from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import re
import mat73


def plot_loss(data_string):
    data = scipy.io.loadmat(data_string)
    training_loss = list(data['training_loss'].T)
    testing_loss = list(data['testing_loss'].T)

    fig, ax1 = plt.subplots()

    colour1 = 'tab:red'
    ax1.set_title('Testing and training loss')
    ax1.set_xlabel('Epochs')
    ax1.plot(np.arange(0, len(training_loss)), training_loss, label='training loss', color=colour1)
    ax1.tick_params(axis='y', labelcolor=colour1)
    ax1.set_ylabel('Training Loss')

    ax2 = ax1.twinx()
    colour2 = 'tab:blue'
    ax2.plot(np.arange(0, len(testing_loss)), testing_loss, label='testing loss', color=colour2)
    ax2.set_ylabel('Testing Loss', color=colour2)
    ax2.tick_params(axis='y', labelcolor=colour2)
    fig.tight_layout()
    plt.savefig(re.search(r'.*\\', str(file)).group() + 'loss_plot.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.show()


file = r'C:\Joe Evans\University\Computing\Summer Project\DeepLearningForJoe\NeuralNet\Datasets\Gaussian\Gaussian_20_1\loss_3D_UNet3.mat'
plot_loss(file)
