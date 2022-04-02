import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras import Model
import json
from numpy import loadtxt
import time


def loading_dataECN(dataset_name, name_expansion):
    start_time = time.time()
    print("Extracting the Data")
    loaded_arr = loadtxt('data_csv/' + dataset_name +'_points_' + name_expansion + '.csv', delimiter=',')
    points = loaded_arr.reshape((loaded_arr.shape[0],200,3))
    print(f"Points shape: {points.shape}")
    loaded_arr = loadtxt('data_csv/' + dataset_name + '_features_' + name_expansion + '.csv', delimiter=',')
    features = loaded_arr.reshape((loaded_arr.shape[0],200,5))
    print(f"Features shape: {features.shape}")
    loaded_arr = loadtxt('data_csv/' + dataset_name + '_mask_' + name_expansion + '.csv', delimiter=',')
    masks = loaded_arr.reshape((loaded_arr.shape[0],200,1))
    print(f"Mask shape: {masks.shape}")
    loaded_arr = loadtxt('data_csv/' + dataset_name + '_labels_' + name_expansion + '.csv', delimiter=',')
    labels = loaded_arr.reshape((loaded_arr.shape[0]))
    print(f"Labels shape: {labels.shape}")
    print("Time spent on extracting " + dataset_name + "dataset: %s seconds" % (time.time() - start_time))
    return points, features, masks, labels

def importCNN(name_extension):
    model_CNN = keras.models.load_model('warmup/models/CNN200k' + name_extension)

    with open("path.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    n_train = 200000
    n_val = 20000
    n_test = 100000

    with h5py.File(jsonObject["CLUSTER_PATH_QCD"], "r") as f:
        images_qcd_test = f["images"][n_train + n_val : n_train + n_test + n_val]
    labels_qcd_test = np.zeros((n_test))

    with h5py.File(jsonObject["CLUSTER_PATH_TOP"], "r") as f:
        images_top_test = f["images"][n_train + n_val : n_train + n_test + n_val]
    labels_top_test = np.ones((n_test))

    images_test = np.concatenate([images_qcd_test, images_top_test])

    labels_test = np.concatenate([labels_qcd_test, labels_top_test])

    x_data_test_CNN = np.reshape(images_test, (n_test * 2, 40,40,1)) 
    y_data_test = tf.keras.utils.to_categorical(labels_test, num_classes=2)

    y_predict_CNN = model_CNN.predict(x_data_test_CNN)

    fpr_keras_CNN, tpr_keras_CNN, thresholds_keras_CNN = roc_curve(y_data_test.argmax(1), y_predict_CNN[:,1])
    auc_keras_CNN = auc(fpr_keras_CNN, tpr_keras_CNN)
    return fpr_keras_CNN, tpr_keras_CNN, auc_keras_CNN


def importLundNet(points, features, mask, labels, model_number):
    model_LundNet = keras.models.load_model('edgeConvNetResults/models/LundNetTF400_' + model_number)
    print("LundNet EVALUATION")
    y_data_test = tf.keras.utils.to_categorical(labels, num_classes=2)

    y_predict_LundNet = model_LundNet.predict([points, features, mask])

    fpr_keras_LundNet, tpr_keras_LundNet, thresholds_keras_LundNet = roc_curve(y_data_test.argmax(1), y_predict_LundNet[:,1])
    auc_keras_LundNet = auc(fpr_keras_LundNet, tpr_keras_LundNet)
    return fpr_keras_LundNet, tpr_keras_LundNet, auc_keras_LundNet

def importNeighboursNet(points, features, mask, labels, model_number):
    model_NeighboursNet = keras.models.load_model('edgeConvNetResults/models/NeighboursNet400_' + model_number)
    print("NeighboursNet EVALUATION")
    y_data_test = tf.keras.utils.to_categorical(labels, num_classes=2)

    y_predict_NeighboursNet = model_NeighboursNet.predict([points, features, mask])

    fpr_keras_NeighboursNet, tpr_keras_NeighboursNet, thresholds_keras_NeighboursNet = roc_curve(y_data_test.argmax(1), y_predict_NeighboursNet[:,1])
    auc_keras_NeighboursNet = auc(fpr_keras_NeighboursNet, tpr_keras_NeighboursNet)
    return fpr_keras_NeighboursNet, tpr_keras_NeighboursNet, auc_keras_NeighboursNet

def importNeighboursNetLite(points, features, mask, labels, model_number):
    model_NeighboursNet = keras.models.load_model('edgeConvNetResults/models/NeighboursNetLite400_' + model_number)
    print("NeighboursNetLite EVALUATION")
    y_data_test = tf.keras.utils.to_categorical(labels, num_classes=2)

    y_predict_NeighboursNet = model_NeighboursNet.predict([points, features, mask])

    fpr_keras_NeighboursNet, tpr_keras_NeighboursNet, thresholds_keras_NeighboursNet = roc_curve(y_data_test.argmax(1), y_predict_NeighboursNet[:,1])
    auc_keras_NeighboursNet = auc(fpr_keras_NeighboursNet, tpr_keras_NeighboursNet)
    return fpr_keras_NeighboursNet, tpr_keras_NeighboursNet, auc_keras_NeighboursNet

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def getBckRejection(tpr_keras, fpr_keras):
    nearestValue30 = find_nearest(tpr_keras, 0.3)
    idx30 = np.where(tpr_keras == nearestValue30)
    e30 = np.mean(1/fpr_keras[idx30])

    nearestValue50 = find_nearest(tpr_keras, 0.5)
    idx50 = np.where(tpr_keras == nearestValue50)
    e50 = np.mean(1/fpr_keras[idx50])
    return e30, e50

print("Start")
favorite_color = pickle.load(open("results_lund_400k/test_ROC_data.pickle", "rb" ) )
sig = favorite_color["signal_eff"]
bkg = favorite_color["background_eff"]
trsh = favorite_color["thresholds"]

fpr_keras_CNN, tpr_keras_CNN, auc_keras_CNN = importCNN("4")
e30_CNN, e50_CNN = getBckRejection(tpr_keras_CNN, fpr_keras_CNN)
print(f"CNN2 e30: {e30_CNN}, e50: {e50_CNN}, auc: {auc_keras_CNN}")

points, features, mask, labels = loading_dataECN("test", "80000")

fpr_keras_LundNet, tpr_keras_LundNet, auc_keras_LundNet = importLundNet(points, features, mask, labels, "1")
e30_LundNet, e50_LundNet = getBckRejection(tpr_keras_LundNet, fpr_keras_LundNet)
print(f"LundNet e30: {e30_LundNet}, e50: {e50_LundNet}, auc = {auc_keras_LundNet}")

fpr_keras_NeighboursNet, tpr_keras_NeighboursNet, auc_keras_NeighboursNet = importNeighboursNet(points, features, mask, labels, "1")
e30_NeighboursNet, e50_NeighboursNet = getBckRejection(tpr_keras_NeighboursNet, fpr_keras_NeighboursNet)
print(f"NeighboursNet e30: {e30_NeighboursNet}, e50: {e50_NeighboursNet}, auc = {auc_keras_NeighboursNet}")

fpr_keras_NeighboursNetLite, tpr_keras_NeighboursNetLite, auc_keras_NeighboursNetLite = importNeighboursNetLite(points, features, mask, labels, "1")
e30_NeighboursNetLite, e50_NeighboursNetLite = getBckRejection(tpr_keras_NeighboursNetLite, fpr_keras_NeighboursNetLite)
print(f"NeighboursNetLite e30: {e30_NeighboursNetLite}, e50: {e50_NeighboursNetLite}, auc = {auc_keras_NeighboursNetLite}")

plt.figure(1)

loaded_ParticleNet = loadtxt('dataFromLiterature/ParticleNet.csv', delimiter=',')
literature_x = []
literature_y = []
for a, b in loaded_ParticleNet:
    literature_x.append(a)
    literature_y.append(b)

plt.plot(sig, 1/(1 - bkg), label='Lundnet5')
plt.plot(tpr_keras_CNN, 1/fpr_keras_CNN, label='CNN')
plt.plot(tpr_keras_LundNet, 1/fpr_keras_LundNet, label='LundNetTF')
plt.plot(tpr_keras_NeighboursNet, 1/fpr_keras_NeighboursNet, label='NeighboursNet')
plt.plot(tpr_keras_NeighboursNetLite, 1/fpr_keras_NeighboursNetLite, label='NeighboursNetLite ')
plt.plot(literature_x,literature_y, color = "gray", linestyle='--', label='ParticleNet (Literature)')
plt.yscale("log")
plt.xlabel(r"$\epsilon_{{s}}$")
plt.ylabel(r"1/$\epsilon_{{b}}$")
plt.grid()
plt.title('ROC curve inverse')
plt.legend(loc='best')
plt.ylim([0,5000])
plt.savefig("edgeConvNetResults/roc200.png")
