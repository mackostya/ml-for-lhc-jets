"""
Implementation of a simple convolutional neural network (CNN) to classify QCD- and top-jets.
"""

import tensorflow as tf
from tensorflow import keras
import h5py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pylab as pylab

models = keras.models
layers = keras.layers


def cnn_model():
    model = models.Sequential(
        [
            layers.Conv2D(
                128,
                input_shape=(40, 40, 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                dilation_rate=1,
                activation="relu",
            ),
            layers.MaxPooling2D((2, 2), strides=None, padding="valid"),
            layers.Conv2D(
                128,
                input_shape=(20, 20, 128),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                dilation_rate=1,
                activation="relu",
            ),
            layers.MaxPooling2D((2, 2), strides=None, padding="valid"),
            layers.Conv2D(
                128,
                input_shape=(10, 10, 128),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                dilation_rate=1,
                activation="relu",
            ),
            layers.Flatten(),
            layers.Dense(36),
            layers.Activation("relu"),
            layers.Dense(6),
            layers.Activation("relu"),
            layers.Dense(2),
            layers.Activation("softmax"),
        ]
    )
    return model


if __name__ == "__main__":

    with open("path.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    n_train = 200000
    n_test = 20000
    n_val = 20000

    with h5py.File(jsonObject["CLUSTER_PATH_QCD"], "r") as f:
        images_qcd_train = f["images"][:n_train]
        images_qcd_val = f["images"][n_train : n_train + n_val]
        images_qcd_test = f["images"][n_train + n_val : n_train + n_test + n_val]
    labels_qcd_train = np.zeros((n_train))
    labels_qcd_val = np.zeros((n_val))
    labels_qcd_test = np.zeros((n_test))

    with h5py.File(jsonObject["CLUSTER_PATH_TOP"], "r") as f:
        images_top_train = f["images"][:n_train]
        images_top_val = f["images"][n_train : n_train + n_val]
        images_top_test = f["images"][n_train + n_val : n_train + n_test + n_val]
    labels_top_train = np.ones((n_train))
    labels_top_val = np.ones((n_val))
    labels_top_test = np.ones((n_test))

    images_train = np.concatenate([images_qcd_train, images_top_train])
    images_val = np.concatenate([images_qcd_val, images_top_val])
    images_test = np.concatenate([images_qcd_test, images_top_test])

    labels_train = np.concatenate([labels_qcd_train, labels_top_train])
    labels_val = np.concatenate([labels_qcd_val, labels_top_val])
    labels_test = np.concatenate([labels_qcd_test, labels_top_test])

    x_data_train = np.reshape(images_train, (n_train * 2, 40, 40, 1))
    x_data_val = np.reshape(images_val, (n_val * 2, 40, 40, 1))
    x_data_test = np.reshape(images_test, (n_test * 2, 40, 40, 1))

    y_data_train = tf.keras.utils.to_categorical(labels_train, num_classes=2)
    y_data_val = tf.keras.utils.to_categorical(labels_val, num_classes=2)
    y_data_test = tf.keras.utils.to_categorical(labels_test, num_classes=2)

    # make output directory
    folder = "results/CNN/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    val_accuracy = []
    val_loss = []
    train_loss = []
    train_accuracy = []

    model = cnn_model()

    print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=0.000001)

    model.fit(
        x_data_train,
        y_data_train,
        epochs=50,
        verbose=2,
        validation_data=(x_data_val, y_data_val),
        batch_size=512,
        shuffle=True,
        callbacks=[reduce_lr, keras.callbacks.EarlyStopping(patience=10)],
    )

    model = keras.models.load_model("models/CNN200k4")

    print()
    print("                             Loss Accuracy for a test")
    print()

    model.evaluate(x_data_test, y_data_test, batch_size=1000)

    # with open('results/historyCNN200k1.csv') as file:
    #         csv_reader = csv.reader(file, delimiter=',')
    #         eliminator = 0
    #         for row in csv_reader:
    #             if eliminator!=0:
    #                 train_accuracy.append(float(row[1]))
    #                 train_loss.append(float(row[2]))
    #                 val_accuracy.append(float(row[4]))
    #                 val_loss.append(float(row[5]))
    #             eliminator = eliminator + 1

    # params = {'legend.fontsize': 25,
    #           'figure.figsize': (15, 5),
    #          'axes.labelsize': 25,
    #          'axes.titlesize':25,
    #          'xtick.labelsize':25,
    #          'ytick.labelsize':25}
    # pylab.rcParams.update(params)

    # fig, axs = plt.subplots(2,1, figsize = (20,20))

    # ax1 = axs[0]
    # ax2 = axs[1]

    # ax1.set_title('Loss')
    # ax2.set_title('Accuracy')
    # ax1.set_xlabel('Epoch')
    # ax2.set_xlabel('Epoch')

    # ax1.plot(train_loss, label = "train_loss")
    # ax2.plot(train_accuracy, label = "train_accuracy")
    # ax1.plot(val_loss, label = "val_loss")
    # ax2.plot(val_accuracy, label = "val_accuracy")
    # ax1.legend()
    # ax1.grid()
    # ax2.legend()
    # ax2.grid()
    # fig.savefig("CNN_loss_acc200k.png")
    # plt.close("all")
