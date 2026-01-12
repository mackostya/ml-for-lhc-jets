import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import loadtxt
import csv
import matplotlib.pyplot as plt
import time
from keras.callbacks import ReduceLROnPlateau

home_folder = "git/bt/"


def edgeFuncAlt(num_points, points, features):
    with tf.name_scope("edgeFunc"):
        # 3 is a number of edges
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        expanded_fts = tf.tile(tf.expand_dims(features, axis=2), (1, 1, 3, 1))  # (2,11,2) -> (2,11,3,2)
        batches = tf.reshape(tf.range(0, batch_size, 1, dtype="int32"), (batch_size, 1))
        ones = tf.ones(shape=(batch_size, num_points * 3), dtype="int32")  # (2, 11 * 3)
        mult = batches * ones  # (2,33)
        reshaped_batches = tf.reshape(mult, [-1, 1])  # (66,1)
        indexes = tf.cast(tf.reshape(points, (batch_size * num_points * 3, 1)), dtype="int32")
        newPositions = tf.concat([reshaped_batches, indexes], 1)
        after_gather = tf.gather_nd(
            features, newPositions
        )  # features shape = (2,11,2) newPositions shape = (2*11*3, 1)
        after_gather = tf.reshape(after_gather, tf.shape(expanded_fts))  # (2,11,3,2)
        diff = expanded_fts - after_gather
        output = tf.concat([expanded_fts, diff], axis=-1)  # (2,11,3,4)
        return output


def edge_conv(
    points, features, num_points, K, channels, with_bn=True, activation="relu", pooling="average", name="edgeconv"
):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    with tf.name_scope("edgeconv"):

        fts = features
        x = edgeFuncAlt(num_points, points, fts)  # (N, P, 3, 2*C)

        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(
                channel,
                kernel_size=(1, 1),
                strides=1,
                data_format="channels_last",
                use_bias=False if with_bn else True,
                kernel_initializer="glorot_normal",
                name="%s_conv%d" % (name, idx),
            )(x)
            if with_bn:
                x = keras.layers.BatchNormalization(name="%s_bn%d" % (name, idx))(x)
            if activation:
                x = keras.layers.Activation(activation, name="%s_act%d" % (name, idx))(x)

        if pooling == "max":
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = keras.layers.Conv2D(
            channels[-1],
            kernel_size=(1, 1),
            strides=1,
            data_format="channels_last",
            use_bias=False if with_bn else True,
            kernel_initializer="glorot_normal",
            name="%s_sc_conv" % name,
        )(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name="%s_sc_bn" % name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if activation:
            return keras.layers.Activation(activation, name="%s_sc_act" % name)(sc + fts)  # (N, P, C')
        else:
            return sc + fts


def _particle_net_base(points, features=None, mask=None, setting=None, name="particle_net"):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    with tf.name_scope(name):
        if features is None:
            features = points
        if mask is not None:
            mask = tf.cast(tf.not_equal(mask, 0), dtype="float32")  # 1 if valid
            coord_shift = tf.multiply(
                199.0, tf.cast(tf.equal(mask, 0), dtype="float32")
            )  # make non-valid positions to 200

        pts = tf.add(coord_shift, points)
        fts = tf.squeeze(
            keras.layers.BatchNormalization(name="%s_fts_bn" % name)(tf.expand_dims(features, axis=2)), axis=2
        )
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            fts = edge_conv(
                pts,
                fts,
                setting.num_points,
                K,
                channels,
                with_bn=True,
                activation="relu",
                pooling=setting.conv_pooling,
                name="%s_%s%d" % (name, "EdgeConv", layer_idx),
            )

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
                x = keras.layers.Dense(units, activation="relu")(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)
            out = keras.layers.Dense(setting.num_class, activation="softmax")(x)
            return out  # (N, num_classes)
        else:
            return pool


class _DotDict:
    pass


def get_particle_net(num_classes, input_shapes):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (3, (64, 64, 64)),
        (3, (128, 128, 128)),
        (3, (256, 256, 256)),
    ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = "average"
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(256, 0.1)]
    setting.num_points = input_shapes["points"][0]

    points = keras.Input(name="points", shape=input_shapes["points"])
    features = keras.Input(name="features", shape=input_shapes["features"]) if "features" in input_shapes else None
    mask = keras.Input(name="mask", shape=input_shapes["mask"]) if "mask" in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name="ParticleNet")

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name="ParticleNet")


def get_lund_net(num_classes, input_shapes):
    r"""
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (3, (32, 32)),
        (3, (32, 32)),
        (3, (64, 64)),
        (3, (64, 64)),
        (3, (128, 128)),
        (3, (128, 128)),
    ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = "average"
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(256, 0.1)]
    setting.num_points = input_shapes["points"][0]

    points = keras.Input(name="points", shape=input_shapes["points"])
    features = keras.Input(name="features", shape=input_shapes["features"]) if "features" in input_shapes else None
    mask = keras.Input(name="mask", shape=input_shapes["mask"]) if "mask" in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name="LundNet")

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name="LundNet")


def get_particle_net_lite(num_classes, input_shapes):
    r"""ParticleNet-Lite model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (3, (32, 32, 32)),
        (3, (64, 64, 64)),
    ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = "average"
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(128, 0.1)]
    setting.num_points = input_shapes["points"][0]

    points = keras.Input(name="points", shape=input_shapes["points"])
    features = keras.Input(name="features", shape=input_shapes["features"]) if "features" in input_shapes else None
    mask = keras.Input(name="mask", shape=input_shapes["mask"]) if "mask" in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name="ParticleNet")

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name="ParticleNet")


def loading_data(dataset_name, name_expansion):
    start_time = time.time()
    print("Extracting the Data")
    loaded_arr = loadtxt(
        home_folder + "data_csv/" + dataset_name + "_points_" + name_expansion + ".csv", delimiter=","
    )
    points = loaded_arr.reshape((loaded_arr.shape[0], 200, 3))
    print(f"Points shape: {points.shape}")
    loaded_arr = loadtxt(
        home_folder + "data_csv/" + dataset_name + "_features_" + name_expansion + ".csv", delimiter=","
    )
    features = loaded_arr.reshape((loaded_arr.shape[0], 200, 5))
    print(f"Features shape: {features.shape}")
    loaded_arr = loadtxt(home_folder + "data_csv/" + dataset_name + "_mask_" + name_expansion + ".csv", delimiter=",")
    masks = loaded_arr.reshape((loaded_arr.shape[0], 200, 1))
    print(f"Mask shape: {masks.shape}")
    loaded_arr = loadtxt(
        home_folder + "data_csv/" + dataset_name + "_labels_" + name_expansion + ".csv", delimiter=","
    )
    labels = loaded_arr.reshape((loaded_arr.shape[0]))
    print(f"Labels shape: {labels.shape}")
    print("Time spent on extracting " + dataset_name + "dataset: %s seconds" % (time.time() - start_time))
    return points, features, masks, labels


if __name__ == "__main__":

    name_of_model = "1"

    num_points = 200
    shapes = {"points": (num_points, 3), "features": (num_points, 5), "mask": (num_points, 1)}
    model = get_particle_net(2, shapes)
    model.summary()

    #### importing data for training and validation
    points, features, mask, labels = loading_data("train", "200000")
    points_val, features_val, mask_val, labels_val = loading_data("val", "20000")

    y_data = tf.keras.utils.to_categorical(labels, num_classes=2)
    y_data_val = tf.keras.utils.to_categorical(labels_val, num_classes=2)

    ##### training the model
    model.compile(
        loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
    )

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.000001)

    model.fit(
        [points, features, mask],
        y_data,
        epochs=50,
        shuffle=True,
        batch_size=512,
        validation_data=([points_val, features_val, mask_val], y_data_val),
        callbacks=[
            reduce_lr,
            keras.callbacks.EarlyStopping(patience=10),
            keras.callbacks.CSVLogger(
                home_folder + "edgeConvNetResults/historyParticleNet200_" + name_of_model + "_.csv"
            ),
        ],
    )
    model.save(home_folder + "edgeConvNetResults/models/ParticleNet200_" + name_of_model)
    print()
    print("                             Loss Accuracy for a test")
    print()

    points_test, features_test, mask_test, labels_test = loading_data("test", "40000")
    y_data_test = tf.keras.utils.to_categorical(labels_test, num_classes=2)

    model.evaluate([points_test, features_test, mask_test], y_data_test, batch_size=512)
