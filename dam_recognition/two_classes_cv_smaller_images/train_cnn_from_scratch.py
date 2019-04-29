import numpy as np
import tensorflow as tf
from tqdm import tqdm

import os
import matplotlib.pyplot as plt

path_upwards = '../../'
import sys
sys.path.extend([path_upwards + '../TailingDamDetection/'])

import config
from nn_architecture.aka_lenet import cnn_adjust_lr
from utils.ml_functions import softmax

rseed = 1000
np.random.seed(rseed)
tf.random.set_random_seed(rseed)

experiment_name = 'two_classes_cv_smaller_images'

path_to_data = path_upwards + config.data_path + config.experiment_data + experiment_name + '/'
path_to_results = path_upwards + config.result_path + config.experiment_data + experiment_name + '/'
if not os.path.exists(path_to_results):
    os.makedirs(path_to_results)

n_epoch = 40
batch_size = 32
n_cv_runs = 10

for cv_run in tqdm(range(n_cv_runs)):
    data = np.load(path_to_data + 'train_test_data_{}.npz'.format(cv_run))

    train_images = data['train_images']
    train_labels = data['train_labels']

    test_images = data['test_images']
    test_labels = data['test_labels']

    cnn = cnn_adjust_lr(n_classes=train_labels.shape[1], input_shape=train_images[0].shape, lr=1e-4)

    train_accuracy = np.zeros((n_epoch,), dtype=np.float64)
    test_accuracy = np.zeros((n_epoch,), dtype=np.float64)

    for epoch in range(n_epoch):
        print('\tepoch %d...' % epoch)

        cnn.fit(train_images, train_labels, epochs=1, shuffle=True, batch_size=batch_size, verbose=0)

        train_prediction = cnn.predict(train_images)
        train_accuracy[epoch] = np.mean(np.argmax(train_prediction, axis=1) == np.argmax(train_labels, axis=1))
        print('\ttrain accuracy {}'.format(train_accuracy[epoch]))

        test_prediction = cnn.predict(test_images)
        test_accuracy[epoch] = np.mean(np.argmax(test_prediction, axis=1) == np.argmax(test_labels, axis=1))
        print('\ttest accuracy {}'.format(test_accuracy[epoch]))

    train_predicted_probs = softmax(train_prediction, axis=1)
    test_predicted_probs = softmax(test_prediction, axis=1)

    plt.plot(range(n_epoch), train_accuracy, 'r--', label='train')
    plt.plot(range(n_epoch), test_accuracy, 'b-', label='test')
    plt.title('Accuracy {} run'.format(cv_run))
    #plt.savefig(path_to_results + 'accuracy_results_{}.pdf'.format(cv_run))
    plt.show()

    if not os.path.exists(path_to_results + 'trained_model/'):
        os.makedirs(path_to_results + 'trained_model/')
    cnn.save(path_to_results + 'trained_model/weights_{}_run.ckpt'.format(cv_run))

    np.savez(path_to_results + 'results_{}_run'.format(cv_run),
             train_accuracy=train_accuracy,
             train_predicted_probs=train_predicted_probs,
             test_accuracy=test_accuracy,
             test_predicted_probs=test_predicted_probs)
