'''
Implementation of a simple autoencoder (AE) to classify QCD- and top-jets.
'''
import h5py
import tensorflow as tf
from tensorflow import keras
from keras import losses
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from keras.layers import Input, Conv2D, PReLU, AveragePooling2D, Flatten, Dense, Reshape, UpSampling2D, MaxPooling2D
from keras import Model
from keras.initializers import Constant
import json

def ae_model():
    bottleneck = 100
    alpha = 0.25
    
    input = Input(shape = (40,40,1))
    encoded = Conv2D(10, 3, padding = "same")(input)
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = Conv2D(10, 3, padding="same")(encoded) 
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = AveragePooling2D()(encoded)
    encoded = Conv2D(10, 3, padding="same")(encoded) 
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = Conv2D(5, 3, padding="same")(encoded) 
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = Conv2D(5, 3, padding="same")(encoded) 
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(400)(encoded)
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = Dense(100)(encoded)
    encoded = PReLU(alpha_initializer=Constant(value=alpha))(encoded)
    encoded = Dense(bottleneck)(encoded)
    encoded = PReLU(name="bneck")(encoded)

    decoded = Dense(100)(encoded)
    decoded = PReLU(alpha_initializer=Constant(value=alpha))(decoded)
    decoded = Dense(400)(decoded)
    decoded = PReLU(alpha_initializer=Constant(value=alpha))(decoded)
    decoded = Reshape((20,20,1))(decoded)
    decoded = Conv2D(5, 3, padding = "same")(decoded)
    decoded = PReLU(alpha_initializer=Constant(value=alpha))(decoded)
    decoded = Conv2D(5, 3, padding = "same")(decoded)
    decoded = PReLU(alpha_initializer=Constant(value=alpha))(decoded)
    decoded = UpSampling2D()(decoded)
    decoded = Conv2D(5, 3, padding = "same")(decoded)
    decoded = PReLU(alpha_initializer=Constant(value=alpha))(decoded)
    decoded = Conv2D(10, 3, padding = "same")(decoded)
    decoded = PReLU(alpha_initializer=Constant(value=alpha))(decoded)
    decoded = Conv2D(1, 3, padding = "same")(decoded)

    model = Model(input, decoded, name = "AE_20k")
    return model

if __name__=="__main__":

    with open("path.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    n = 20000

    with h5py.File(jsonObject["CLUSTER_PATH_QCD"], "r") as f:
        images_qcd = f["images"][:n]

    x_data = np.reshape(images_qcd, (20000,40,40,1)) 

    # make output directory
    folder = 'results/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    model = ae_model()

    model.compile(optimizer = "adam", 
                    loss = losses.MeanSquaredError())
    model.fit(x_data, x_data,
                    epochs = 30,
                    validation_split = 0.2,
                    verbose = 2,
                    callbacks= [keras.callbacks.EarlyStopping(patience = 3),keras.callbacks.CSVLogger(folder + 'historyAE_small_data_20k.csv')])

    model.save("models/AE_20k")

    ################################################################################
    ##################################### PLOT #####################################
    ################################################################################

    val_loss = []
    loss = []

    with open('results/historyAE_small_data_20k.csv') as file:
            csv_reader = csv.reader(file, delimiter=',')
            eliminator = 0
            for row in csv_reader:
                if eliminator!=0:    
                    loss.append(float(row[1]))
                    val_loss.append(float(row[2]))
                eliminator = eliminator + 1

    fig = plt.figure()

    plt.title('Loss')
    plt.xlabel('Epoch')

    plt.plot(loss, label = "loss")
    plt.plot(val_loss, label = "val_loss")
    plt.legend()
    plt.grid()
    fig.savefig("AE_images/AE_loss_small_data_20k.png")
    plt.close("all")
