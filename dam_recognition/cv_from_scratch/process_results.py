import numpy as np
import matplotlib.pyplot as plt

import config

path_upwards = '../../'
experiment_name = 'cv_10_folds'

path_to_results = path_upwards + config.result_path + config.experiment_data + experiment_name + '/'

n_runs = 10
n_epoch = 100

all_train_accuracy = np.zeros((n_runs, n_epoch), dtype=np.float64)
all_test_accuracy = np.zeros((n_runs, n_epoch), dtype=np.float64)

for run in range(n_runs):
    cur_results = np.load(path_to_results + 'results_{}_run.npz'.format(run))
    all_train_accuracy[run] = cur_results['train_accuracy']
    all_test_accuracy[run] = cur_results['test_accuracy']

mean_train_accuracy = np.mean(all_train_accuracy, axis=0)
mean_test_accuracy = np.mean(all_test_accuracy, axis=0)

std_train_accuracy = np.std(all_train_accuracy, axis=0)
std_test_accuracy = np.std(all_test_accuracy, axis=0)

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


