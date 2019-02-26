import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import config
from utils.utils_input_output import read_label_dictionary
from utils.utils_plots import plot_confusion_matrix_with_confidence

path_upwards = '../../'
experiment_name = 'cv_10_folds'

path_to_data = path_upwards + config.data_path + config.experiment_data + experiment_name + '/'
path_to_results = path_upwards + config.result_path + config.experiment_data + experiment_name + '/'

n_runs = 10
n_epoch = 100
n_classes = 3

label_dictionary = read_label_dictionary(path_to_data + "label_dictionary.txt")
class_codes = np.arange(n_classes)
class_labels = [label_dictionary[code] for code in class_codes]

all_train_accuracy = np.zeros((n_runs, n_epoch), dtype=np.float64)
all_test_accuracy = np.zeros((n_runs, n_epoch), dtype=np.float64)
all_test_cm = np.zeros((n_runs, n_classes, n_classes), dtype=np.float64)

for run in range(n_runs):
    cur_results = np.load(path_to_results + 'results_{}_run.npz'.format(run))
    cur_data = np.load(path_to_data + 'train_test_data_{}.npz'.format(run))

    all_train_accuracy[run] = cur_results['train_accuracy']
    all_test_accuracy[run] = cur_results['test_accuracy']

    all_test_cm[run] = confusion_matrix(np.argmax(cur_data['test_labels'], axis=1),
                                        np.argmax(cur_results['test_prediction'], axis=1))
    all_test_cm[run] = all_test_cm[run] / all_test_cm[run].sum(axis=1)[:, np.newaxis]


mean_train_accuracy = np.mean(all_train_accuracy, axis=0)
mean_test_accuracy = np.mean(all_test_accuracy, axis=0)
mean_test_cm = np.mean(all_test_cm, axis=0)

std_train_accuracy = np.std(all_train_accuracy, axis=0)
std_test_accuracy = np.std(all_test_accuracy, axis=0)
std_test_cm = np.std(all_test_cm, axis=0)

x = np.arange(n_epoch)

fig, ax = plt.subplots()
plt.plot(x, mean_train_accuracy, 'r--', linewidth=2.0, label='train')
ax.fill_between(x,
                np.maximum(mean_train_accuracy - std_train_accuracy, np.zeros((n_epoch,), dtype=np.float64)),
                np.minimum(mean_train_accuracy + std_train_accuracy, np.ones((n_epoch,), dtype=np.float64)),
                facecolor='r', alpha=0.1)

plt.plot(x, mean_test_accuracy, 'b-', linewidth=2.0, label='test')
ax.fill_between(x,
                np.maximum(mean_test_accuracy - std_test_accuracy, np.zeros((n_epoch,), dtype=np.float64)),
                np.minimum(mean_test_accuracy + std_test_accuracy, np.ones((n_epoch,), dtype=np.float64)),
                facecolor='b', alpha=0.2)
plt.legend()
plt.savefig(path_to_results + 'joint_accuracy_plots.pdf', format='pdf')
plt.show()


plot_confusion_matrix_with_confidence(mean_test_cm, std_test_cm,
                                      classes=class_labels, title='Confusion matrix on test')
plt.savefig(path_to_results + 'joint_cm_plot.pdf', format='pdf')
plt.show()

