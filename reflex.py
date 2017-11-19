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
from keras.callbacks import Callback, TensorBoard
from keras.preprocessing import image
from keras.constraints import max_norm

SEED = 23
MODEL_PATH = "reflex.h5"
np.random.seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


class ReflexDataGenerator(image.ImageDataGenerator):
    '''
    Provides uniform zoom in both directions
    '''
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

    def load_files(self, calculate_class_weights=True):
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
            self.class_weights = self.y.sum(axis=0)
            self.class_weights = self.class_weights.max() / self.class_weights

        X_filter = []
        for idx, path in enumerate(X_paths):
            raw_path = path[:-11] + "png"
            if raw_path in set(y_paths.values):
                X_filter.append(idx)
        self.X = X_full[X_filter]

        logging.info("Normalizing %d images", self.X.shape[0])
        self.X = self.X.reshape(self.X.shape[0], self.img_x, self.img_y, 1)
        self.X = self.X.astype('float32')
        self.X /= 255.0
        self.y = self.y.as_matrix()

    def split_data(self, test_ratio):
        if self.X is None or self.y is None:
            raise Exception("No images! Load image files before splitting the data.")

        logging.info("Splitting %d examples into training and testing sets", self.X.shape[0])
        # Rozwazyc Iterative Stratification: On the Stratification of Multi-Label Data, Tsoumakas et al.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_ratio, random_state=SEED,
                                                            stratify=self.y)

        logging.debug("Training set shape: %s", X_train.shape)
        logging.debug("Testing set shape: %s", X_test.shape)
        logging.debug("Training labels shape: %s", y_train.shape)
        logging.debug("Testing labels shape: %s", y_test.shape)

        return X_train, X_test, y_train, y_test

    def get_cifar_model(self):
        model = Sequential()
                
        #Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape, name="block1_conv1"))
        model.add(Dropout(0.2, name="block1_drop"))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name="block1_conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="block1_pool"))
                
        #Block 2
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv1"))
        model.add(Dropout(0.2, name="block2_drop"))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="block2_pool"))
                
        #Block 3
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv1"))
        model.add(Dropout(0.2, name="block3_drop"))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="block3_pool"))
        
        # Classification block
        model.add(Flatten(name="flatten"))
        model.add(Dropout(0.2, name="clf_drop1"))
        model.add(Dense(1024, activation='relu', kernel_constraint=max_norm(3), name="clf_fc1"))
        model.add(Dropout(0.2, name="clf_drop2"))
        model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3), name="clf_fc2"))
        model.add(Dropout(0.2, name="clf_drop3"))
        model.add(Dense(self.num_classes, activation='sigmoid', name="predictions"))
        
        return model
        
    def get_vgg_model(self):
        model = Sequential()
        
        #Block 1
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=self.input_shape, name="block1_conv1"))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"))

        #Block 2
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1"))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1"))
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2"))
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1"))
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2"))
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"))
        
        # Block 5
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1"))
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2"))
        model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool"))

        # Classification block
        model.add(Flatten(name="flatten"))
        model.add(Dense(512, activation='relu', name="clf_fc1"))
        model.add(Dense(512, activation='relu', name="clf_fc2"))
        #model.add(Dense(4096, activation='relu', name="clf_fc1"))
        #model.add(Dense(4096, activation='relu', name="clf_fc2"))
        model.add(Dense(self.num_classes, activation='sigmoid', name="predictions"))

        return model
        
    def train_model(self, X_train, y_train, X_val, y_val, verbose=1, save_model=True, model_name=MODEL_PATH,
                    plot_learning_curve=True, epochs=10, batch_size=64, vgg=True):
        if vgg:
            model = self.get_vgg_model()
        else:
            model = self.get_cifar_model()
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        train_datagen = ReflexDataGenerator(
            zoom_range= (1, 1.2),
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=45,
        )
        train_iterator = train_datagen.flow(X_train, y_train, batch_size=batch_size, seed=SEED)
        val_iterator = ReflexDataGenerator().flow(X_val, y_val, batch_size=batch_size, seed=SEED)
        tb_callback = TensorBoard(log_dir="./log_" + ("vgg" if vgg else "cifar"), histogram_freq=1, batch_size=batch_size, write_graph=True, write_images=True)

        history = model.fit_generator(train_iterator,
                                      epochs=epochs,
                                      steps_per_epoch=max(len(X_train) / batch_size, 1),
                                      class_weight=self.class_weights,
                                      verbose=verbose,
                                      callbacks=[tb_callback],
                                      validation_data=val_iterator,
                                      validation_steps=max(len(X_val) / batch_size, 1))

        if plot_learning_curve:
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(model_name + ".acc.png")
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(model_name + ".loss.png")
            plt.close()

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
            best_threshold = np.ones(y_proba.shape[1]) * 0.5

        logging.info("Class thresholds: %s", best_threshold)
        y_pred = np.array([[1 if y_proba[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in
                           range(len(y_test))])

        logging.info("Hamming score: %.3f", self._hamming_score(y_test, y_pred))
        logging.info("Exact match ratio: %.3f",
                     metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        logging.info("Hamming loss: %.3f", metrics.hamming_loss(y_test, y_pred))
        logging.info("Micro-averaged precision: %.3f", metrics.precision_score(y_test, y_pred, average="micro"))
        logging.info("Micro-averaged recall: %.3f", metrics.recall_score(y_test, y_pred, average="micro"))
        logging.info("Micro-averaged F-score: %.3f", metrics.f1_score(y_test, y_pred, average="micro"))
        logging.info("Macro-averaged precision: %.3f", metrics.precision_score(y_test, y_pred, average="macro"))
        logging.info("Macro-averaged recall: %.3f", metrics.recall_score(y_test, y_pred, average="macro"))
        logging.info("Macro-averaged F-score: %.3f", metrics.f1_score(y_test, y_pred, average="macro"))

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
    
    reflex.train_model(X_train, y_train, X_val=X_test, y_val=y_test, model_name="reflex_cifar.h5", vgg=False)
    reflex.test_model(X_test, y_test, model_name="reflex_cifar.h5")
    
    reflex.train_model(X_train, y_train, X_val=X_test, y_val=y_test, model_name="reflex_vgg.h5", vgg=True)
    reflex.test_model(X_test, y_test, model_name="reflex_vgg.h5")
