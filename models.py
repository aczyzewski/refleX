# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import max_norm

__author__ = "Dariusz Brzezinski"


class ReflexModel(object):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create(self):
        pass


class DropoutModel(ReflexModel):
    def __init__(self, input_shape, num_classes, max_norm_value=3, dropout_ratio=0.2, use_dropout=True):
        self.max_norm_value = max_norm_value
        self.dropout_ratio = dropout_ratio
        self.use_dropout = use_dropout
        super(DropoutModel, self).__init__(input_shape, num_classes)

    def __repr__(self):
        if self.use_dropout:
            return 'dropout_max=%.2f_drop=%.2f' % (self.max_norm_value, self.dropout_ratio)
        else:
            return 'dropout_max=%.2f' % (self.max_norm_value)

    def create(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name="block1_conv1", input_shape=self.input_shape))
        if self.use_dropout: model.add(Dropout(self.dropout_ratio, name="block1_drop"))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name="block1_conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="block1_pool"))

        # Block 2
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv1"))
        if self.use_dropout: model.add(Dropout(self.dropout_ratio, name="block2_drop"))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="block2_pool"))

        # Block 3
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv1"))
        if self.use_dropout: model.add(Dropout(self.dropout_ratio, name="block3_drop"))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name="block3_conv2"))
        model.add(MaxPooling2D(pool_size=(2, 2), name="block3_pool"))

        # Classification block
        model.add(Flatten(name="flatten"))
        if self.use_dropout: model.add(Dropout(self.dropout_ratio, name="clf_drop1"))
        model.add(Dense(1024, activation='relu', kernel_constraint=max_norm(self.max_norm_value), name="clf_fc1"))
        if self.use_dropout: model.add(Dropout(self.dropout_ratio, name="clf_drop2"))
        model.add(Dense(512, activation='relu', kernel_constraint=max_norm(self.max_norm_value), name="clf_fc2"))
        if self.use_dropout: model.add(Dropout(self.dropout_ratio, name="clf_drop3"))
        model.add(Dense(self.num_classes, activation='sigmoid', name="predictions"))

        return model


class VggModel(ReflexModel):
    def __init__(self, input_shape, num_classes, blocks=5):
        if blocks < 3 or blocks > 5:
            raise Exception("Use from 3 to 5 blocks!")
        self.blocks = blocks
        super(VggModel, self).__init__(input_shape, num_classes)

    def __repr__(self):
        return 'vgg_blocks=%d' % self.blocks

    def create(self):
        model = Sequential()

        # Block 1
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1", input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1"))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1"))
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2"))
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"))

        if self.blocks > 3:
            # Block 4
            model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1"))
            model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2"))
            model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3"))
            model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"))

        if self.blocks > 4:
            # Block 5
            model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1"))
            model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2"))
            model.add(Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3"))
            model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool"))

        # Classification block
        model.add(Flatten(name="flatten"))
        model.add(Dense(1024, activation='relu', name="clf_fc1"))
        model.add(Dense(1024, activation='relu', name="clf_fc2"))
        model.add(Dense(self.num_classes, activation='sigmoid', name="predictions"))

        return model