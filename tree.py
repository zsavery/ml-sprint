import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

np.random.seed(42)

actual = np.random.binomial(1, 0.9, 1000)
predicted = np.random.binomial(1, 0.9, 1000)

conf_matrix = metrics.confusion_matrix(actual, predicted)

conf_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                              display_labels=[True, False])
conf_display.plot()
plt.show()