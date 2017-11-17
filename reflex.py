import glob
import numpy as np
import pandas as pd
import logging

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import misc

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback

SEED = 23
MODEL_PATH = "reflex.h5"
np.random.seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


class Reflex():
    def __init__(self, img_x, img_y, num_classes, image_files, label_file):
        self.img_x = img_x
        self.img_y = img_y
        self.input_shape = (img_x, img_y, 1)
        self.num_classes = num_classes
        self.image_files = image_files
        self.label_file = label_file
        self.X = None
        self.y = None

    def load_files(self):
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

        X_filter = []
        for idx, path in enumerate(X_paths):
            raw_path = path[:-11] + "png"
            if raw_path in set(y_paths.values):
                X_filter.append(idx)
        self.X = X_full[X_filter]

        logging.info("Normalizing %d images", self.X.shape[0])
        self.X = self.X.reshape(self.X.shape[0], self.img_x, self.img_y, 1)
        self.X = self.X.astype('float32')
        self.X /= 255
        self.y = self.y.as_matrix()

    def split_data(self, test_ratio):
        if self.X is None or self.y is None:
            raise Exception("No images! Load image files before splitting the data.")

        logging.info("Splitting %d examples into training and testing sets", self.X.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_ratio, random_state=SEED)

        logging.debug("Training set shape: %s", X_train.shape)
        logging.debug("Testing set shape: %s", X_test.shape)
        logging.debug("Training labels shape: %s", y_train.shape)
        logging.debug("Testing labels shape: %s", y_test.shape)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, verbose=1, save_model=True, model_name=MODEL_PATH,
                    plot_learning_curve=True):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        history = AccuracyHistory()

        model.fit(X_train, y_train,
                  batch_size=128,
                  epochs=10,
                  verbose=verbose,
                  validation_split=0.1,
                  callbacks=[history])

        if plot_learning_curve:
            plt.plot(range(1, 11), history.acc)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.show()

        if save_model:
            logging.info("Saving model to file: %s", model_name)
            model.save(model_name)

        return model

    def test_model(self, X_test, y_test, model_object=None, load_model_from_file=True, model_name=MODEL_PATH,
                   diversify_thresholds=False):
        if load_model_from_file:
            logging.info("Loading model from file: %s", model_name)
            model = load_model(model_name)
        elif model_object is not None:
            model = model_object
        else:
            raise Exception("No model specified! Provide model file or pass the model_object argument")

        y_proba = model.predict_proba(X_test)
        y_proba = np.array(y_proba)
        threshold = np.arange(0.1, 0.9, 0.1)

        if diversify_thresholds:
            acc = []
            accuracies = []
            best_threshold = np.zeros(y_proba.shape[1])
            for i in range(y_proba.shape[1]):
                y_prob = np.array(y_proba[:, i])
                for j in threshold:
                    y_pred = [1 if prob >= j else 0 for prob in y_prob]
                    acc.append(metrics.matthews_corrcoef(y_test[:, i], y_pred))
                acc = np.array(acc)
                index = np.where(acc == acc.max())
                accuracies.append(acc.max())
                best_threshold[i] = threshold[index[0][0]]
                acc = []
        else:
            best_threshold = np.ones(y_proba.shape[1])*0.5

        logging.info("Class thresholds: %s", best_threshold)
        y_pred = np.array([[1 if y_proba[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in
                           range(len(y_test))])

        logging.info("Hamming score: %.3f", self._hamming_score(y_test, y_pred))
        logging.info("Exact match ratio: %.3f", metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        logging.info("Hamming loss: %.3f", metrics.hamming_loss(y_test, y_pred))
        logging.info("Micro-averaged precision: %.3f", metrics.precision_score(y_test, y_pred, average="micro"))
        logging.info("Micro-averaged recall: %.3f", metrics.recall_score(y_test, y_pred, average="micro"))

    def _hamming_score(self, y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])

            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / \
                        float(len(set_true.union(set_pred)))
            acc_list.append(tmp_a)
        return np.mean(acc_list)


if __name__ == "__main__":
    files = [fn for fn in glob.glob("./data/*.png") if (fn.endswith("300x300.png"))]
    reflex = Reflex(img_x=300, img_y=300, num_classes=7, image_files=files, label_file="reflex.csv")
    reflex.load_files()
    X_train, X_test, y_train, y_test = reflex.split_data(test_ratio=0.1)
    # reflex.train_model(X_train, y_train)
    reflex.test_model(X_test, y_test)
