import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc

path_upwards = '../../'
import sys
sys.path.extend([path_upwards + '../TailingDamDetection/'])

import config
from utils.ml_functions import compute_confusion_matrix
from utils.utils_input_output import read_label_dictionary
from utils.utils_plots import plot_confusion_matrix_with_confidence

experiment_name = 'two_classes_cv_images_v3'

path_to_data = path_upwards + config.data_path + config.experiment_data + experiment_name + '/'
path_to_results = path_upwards + config.result_path + config.experiment_data + experiment_name + '/'

n_runs = 10
n_epoch = 40
n_classes = 2

label_dictionary = read_label_dictionary(path_to_data + "label_dictionary.txt")
class_codes = np.arange(n_classes)
class_labels = [label_dictionary[code] for code in class_codes]

all_train_accuracy = np.zeros((n_runs, n_epoch), dtype=np.float64)
all_test_accuracy = np.zeros((n_runs, n_epoch), dtype=np.float64)
all_test_cm = np.zeros((n_runs, n_classes, n_classes), dtype=np.float64)
all_test_auc = np.zeros((n_runs,), dtype=np.float64)

tprs = []
mean_fpr = np.linspace(0, 1, 100)

for run in range(n_runs):
    cur_results = np.load(path_to_results + 'results_{}_run.npz'.format(run))
    cur_data = np.load(path_to_data + 'train_test_data_{}.npz'.format(run))

    all_train_accuracy[run] = cur_results['train_accuracy']
    all_test_accuracy[run] = cur_results['test_accuracy']

    all_test_cm[run] = compute_confusion_matrix(np.argmax(cur_data['test_labels'], axis=1),
                                                np.argmax(cur_results['test_predicted_probs'], axis=1),
                                                normalise=True)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(np.argmax(cur_data['test_labels'], axis=1),
                                     cur_results['test_predicted_probs'][:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    all_test_auc[run] = auc(fpr, tpr)


mean_train_accuracy = np.mean(all_train_accuracy, axis=0)
mean_test_accuracy = np.mean(all_test_accuracy, axis=0)
mean_test_cm = np.mean(all_test_cm, axis=0)
mean_test_auc = np.mean(all_test_auc)

std_train_accuracy = np.std(all_train_accuracy, axis=0)
std_test_accuracy = np.std(all_test_accuracy, axis=0)
std_test_cm = np.std(all_test_cm, axis=0)
std_test_auc = np.std(all_test_auc)

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

# roc-curves
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b',
         linewidth=2)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2)
plt.title('ROC curve')
plt.savefig(path_to_results + 'joint_roc_curve.pdf', format='pdf')
plt.show()
