# coding: utf-8

import glob
import numpy as np
import pandas as pd
import logging
import sys
import getopt
import os
import gc
import time

from sklearn import metrics as sk_metrics
from scipy import misc
import metrics
import util

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.preprocessing import image

import tensorflow as tf
import random as rn

from models import DropoutModel, VggModel, FcModel, PoolingModel

__author__ = "Dariusz Brzezinski"

SEED = 23
MODELS_PATH = "./models/"
LOGS_PATH = "./logs/"
DATA_PATH = "./data/"
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


class ReflexDataGenerator(image.ImageDataGenerator):
    """
    Provides uniform zoom in both directions
    """

    def random_transform(self, x, seed=None):
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zx, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = image.transform_matrix_offset_center(transform_matrix, h, w)
            x = image.apply_transform(x, transform_matrix, img_channel_axis,
                                      fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x,
                                           self.channel_shift_range,
                                           img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = image.flip_axis(x, img_row_axis)

        return x


class Reflex:
    def __init__(self, img_x, img_y, num_classes, image_files, label_file):
        self.img_x = img_x
        self.img_y = img_y
        self.input_shape = (img_x, img_y, 1)
        self.num_classes = num_classes
        self.image_files = image_files
        self.label_file = label_file
        self.X = None
        self.y = None
        self.class_weights = None

    def preprocess_images(self, X):
        logging.info("Normalizing %d images", X.shape[0])
        X = X.reshape(X.shape[0], self.img_x, self.img_y, 1)
        X = X.astype('float32')
        X /= 255.0
        X = 1.0 - X

        return X

    def load_files(self, calculate_class_weights=True):
        gc.collect()
        images = []
        image_paths = []

        logging.info("Reading files")
        for image_path in self.image_files:
            image_paths.append(image_path)
            images.append(misc.imread(image_path, flatten=True))

        X_full, X_paths = np.stack(images), image_paths

        y_df = pd.read_csv(self.label_file)
        y_paths = y_df.iloc[:, 0]
        self.y = y_df.iloc[:, 1:]
        if calculate_class_weights:
            class_weights = self.y.sum(axis=0)
            class_weights = class_weights.max() / class_weights
            weight_dict = {}
            idx = 0
            for cls, weight in class_weights.iteritems():
                weight_dict[idx] = weight
                idx += 1
            self.class_weights = weight_dict

        X_filter = []
        for idx, path in enumerate(X_paths):
            raw_path = path[:-11] + "png"
            if raw_path in set(y_paths.values):
                X_filter.append(idx)
        self.X = X_full[X_filter]
        self.X = self.preprocess_images(self.X)

        self.y = self.y.as_matrix()
        self.y = self.y.astype('float32')

        gc.collect()
        logging.debug("X shape: %s", self.X.shape)
        logging.debug("y shape: %s", self.y.shape)

    def split_data(self, test_ratio):
        if self.X is None or self.y is None:
            raise Exception("No images! Load image files before splitting the data.")

        logging.info("Splitting %d examples into training and testing sets", self.X.shape[0])
        X_train, X_test, y_train, y_test = util.train_test_split(self.X, self.y, test_size=test_ratio, stratify=self.y,
                                                                 random_state=SEED)

        logging.debug("Training set shape: %s", X_train.shape)
        logging.debug("Testing set shape: %s", X_test.shape)
        logging.debug("Training labels shape: %s", y_train.shape)
        logging.debug("Testing labels shape: %s", y_test.shape)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def reset_session():
        K.clear_session()
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(SEED)
        rn.seed(SEED + SEED)
        tf.set_random_seed(SEED + SEED + SEED)
        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)

    def train_model(self, X_train, y_train, X_val, y_val, model, model_name, verbose=1, save_model=True, epochs=50,
                    learning_rate=0.001, batch_size=32, debug=True, augment=False, loss="binary_crossentropy"):
        if debug:
            tb_callback = TensorBoard(log_dir=LOGS_PATH + model_name, batch_size=batch_size, histogram_freq=epochs / 10,
                                      write_graph=True, write_images=False)
        else:
            tb_callback = None

        model.compile(loss=loss, optimizer=Adam(lr=learning_rate),
                      metrics=[metrics.hamming_loss, metrics.exact_match_ratio])

        if augment:
            datagen = ReflexDataGenerator(
                zoom_range=(1, 1.2),
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=45,
            )
            datagen.fit(X_train)
            data_iterator = datagen.flow(X_train, y_train, batch_size=batch_size, seed=SEED)
            model.fit_generator(data_iterator,
                                epochs=epochs,
                                steps_per_epoch=max(len(X_train) / batch_size, 1),
                                class_weight=self.class_weights,
                                verbose=verbose,
                                callbacks=[tb_callback],
                                validation_data=(X_val, y_val),
                                validation_steps=max(len(X_val) / batch_size, 1))
        else:
            model.fit(X_train,
                      y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      class_weight=self.class_weights,
                      verbose=verbose,
                      callbacks=[tb_callback],
                      validation_data=(X_val, y_val))

        if save_model:
            logging.info("Saving model to file: %s", model_name)
            if not os.path.exists(MODELS_PATH):
                os.mkdir(MODELS_PATH)
            model.save(MODELS_PATH + model_name)

        return model

    def test_model(self, X_test, y_test, model_object=None, load_model_from_file=True, model_name=None):
        if load_model_from_file:
            logging.info("Loading model from file: %s", model_name)
            model = load_model(MODELS_PATH + model_name,
                               custom_objects={
                                   'hamming_loss': metrics.hamming_loss,
                                   'exact_match_ratio': metrics.exact_match_ratio
                               })
        elif model_object is not None:
            model = model_object
        else:
            raise Exception("No model specified! Provide model file or pass the model_object argument")

        y_proba = model.predict_proba(X_test)
        y_proba = np.array(y_proba)
        best_threshold = np.ones(self.num_classes) * 0.5

        logging.info("Class thresholds: %s", best_threshold)
        y_pred = np.array([[1 if y_proba[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in
                           range(len(y_test))])

        logging.info("Exact match ratio: %.3f", sk_metrics.accuracy_score(y_test, y_pred))
        logging.info("Hamming score: %.3f", metrics.hamming_score(y_test, y_pred))
        logging.info("Hamming loss: %.3f", sk_metrics.hamming_loss(y_test, y_pred))
        logging.info("Micro-averaged precision: %.3f", sk_metrics.precision_score(y_test, y_pred, average="micro"))
        logging.info("Micro-averaged recall: %.3f", sk_metrics.recall_score(y_test, y_pred, average="micro"))
        logging.info("Micro-averaged F-score: %.3f", sk_metrics.f1_score(y_test, y_pred, average="micro"))
        logging.info("Macro-averaged precision: %.3f", sk_metrics.precision_score(y_test, y_pred, average="macro"))
        logging.info("Macro-averaged recall: %.3f", sk_metrics.recall_score(y_test, y_pred, average="macro"))
        logging.info("Macro-averaged F-score: %.3f", sk_metrics.f1_score(y_test, y_pred, average="macro"))


def get_run_name(model, resolution, learning_rate, epochs, augment, batch):
    return str(model) + "_res=" + str(resolution) + "_lr=" + str(learning_rate) + "_ep=" + str(epochs) + "_aug=" + \
           str(augment) + "_b=" + str(batch) + "_t=" + time.strftime("%Y%m%d_%H%M%S", time.localtime())


def run_experiments(res, num_classes, models, lrs, epochs, augmenting, batch_sizes, test_ratio, weights):
    files = [fn for fn in glob.glob(DATA_PATH + "*.png") if (fn.endswith(res + "x" + res + ".png"))]
    reflex = Reflex(int(res), int(res), num_classes=num_classes, image_files=files, label_file="reflex.csv")
    reflex.load_files(calculate_class_weights=weights)
    X_train, X_test, y_train, y_test = reflex.split_data(test_ratio=test_ratio)

    # import keras
    # from keras.datasets import mnist
    #
    # res = 28
    # num_classes = 10
    # epochs = 10
    # reflex = Reflex(res, res, num_classes, None, None)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # X_train = x_train.reshape(x_train.shape[0], res, res, 1)
    # X_test = x_test.reshape(x_test.shape[0], res, res, 1)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # reflex.input_shape = (28, 28, 1)
    # models = [
    #     DropoutModel(reflex.input_shape, num_classes, "sigmoid", dropout_ratio=0.2)
    # ]

    for model in models:
        for lr in lrs:
            for augment in augmenting:
                for batch_size in batch_sizes:
                    name = get_run_name(model, res, lr, epochs, augment, batch_size)
                    reflex.reset_session()
                    reflex.train_model(X_train, y_train, X_test, y_test, model.create(), name, epochs=epochs,
                                       learning_rate=lr, augment=augment, batch_size=batch_size,
                                       loss="binary_crossentropy")
                    reflex.test_model(X_test, y_test, model_name=name)


if __name__ == "__main__":
    usage = "Usage: python reflex.py -r <int resolution>\n" \
            "       python reflex.py --resolution <int resolution>"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "r:", ["resolution="])
        if len(opts) == 1 and len(args) == 0 and opts[0][0] in ("-r", "--resolution"):
            resolution = opts[0][1]
            input_shape = (int(resolution), int(resolution), 1)
            num_classes = 7
            models = [
                # DropoutModel(input_shape, num_classes, use_dropout=False, activation="sigmoid"),
                #DropoutModel(input_shape, num_classes, dropout_ratio=0.4),
                # VggModel(input_shape, num_classes, "sigmoid", 5, use_dropout=False, dropout_ratio=0.2),
                PoolingModel(input_shape, num_classes, activation="sigmoid"),
                # FcModel(input_shape, num_classes)
            ]
            lrs = [0.001]
            epochs = 200
            batch_sizes = [32]
            augmenting = [False]
            test_ratio = 0.3
            weights = False

            run_experiments(resolution, num_classes, models, lrs, epochs, augmenting, batch_sizes, test_ratio, weights)
        else:
            print(usage)
            sys.exit(2)
    except getopt.GetoptError as err:
        print(str(err))
        print(usage)
        sys.exit(2)
