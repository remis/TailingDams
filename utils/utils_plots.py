import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix_with_confidence(cm_mean, cm_std, classes,
                                          title='Confusion matrix',
                                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """

    plt.figure(figsize=(8, 8))
    plt.imshow(cm_mean, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm_mean.max() / 2.
    for i, j in itertools.product(range(cm_mean.shape[0]), range(cm_mean.shape[1])):
        plt.text(j, i, r'{:.3f} $\pm$ {:.3f}'.format(cm_mean[i, j], cm_std[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_mean[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
