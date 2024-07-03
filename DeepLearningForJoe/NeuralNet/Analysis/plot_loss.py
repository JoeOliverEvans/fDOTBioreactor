from matplotlib import pyplot as plt
import numpy as np
import mat73


def plot_loss(data_string):
    data = mat73.loadmat(data_string)
    training_loss = data['training_loss']
    testing_loss = data['testing_loss']

    plt.plot(x=np.linspace(0, len(training_loss)), y=training_loss, label='training loss', color='red')
    plt.plot(x=np.linspace(0, len(testing_loss)), y=testing_loss, label='testing loss', color='blue')
    plt.title('Testing and training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


file = r'C:\Joe Evans\University\Computing\Summer Project\DeepLearningForJoe\NeuralNet\_Dummy\test_processed.mat'
