from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, InputLayer, MaxPooling2D
import keras
import tensorflow as tf


def keras_cross_entropy_with_logits(y_true, y_pred):
    return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)


def tf_cross_entropy_with_logits(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def build_model_without_last_layer(base_model):
    model_without_last_layer = Model(inputs=base_model.input,
                                     outputs=base_model.layers[-2].output)

    return model_without_last_layer


def cnn_adjust_lr(n_classes, input_shape=(28, 28, 1), lr=1e-4):
    cnn_model = Sequential()
    cnn_model.add(InputLayer(input_shape=input_shape))
    cnn_model.add(Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1024))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(n_classes))

    adam = keras.optimizers.Adam(lr=lr)

    cnn_model.compile(optimizer=adam, loss=tf_cross_entropy_with_logits, metrics=['accuracy'])

    return cnn_model
