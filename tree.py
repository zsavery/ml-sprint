import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

if __name__ == '__main__':
    # grab random seed
    np.random.seed(42)
    # create random data
    actual = np.random.binomial(1, 0.9, 1000)
    predicted = np.random.binomial(1, 0.9, 1000)
    # n - number of runs
    # p - percentage

    # create confusion matrix
    conf_matrix = metrics.confusion_matrix(actual, predicted)

    # Display confusion matrix
    conf_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[True, False])
    conf_display.plot() # doesn't need plt.plot()
    plt.show()