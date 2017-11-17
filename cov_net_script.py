import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from scipy import misc

SEED = 23
np.random.seed(SEED)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback

img_x, img_y = 300, 300
num_classes = 7
file_list = [fn for fn in glob.glob("./data/*.png") if (fn.endswith("300x300.png"))]
images = []
image_paths = []
for image_path in file_list:
    image_paths.append(image_path)
    images.append(misc.imread(image_path, flatten=True))

X_full, X_paths = np.stack(images), image_paths

y_df = pd.read_csv("reflex.csv")
y_paths = y_df.iloc[:, 0]
y = y_df.iloc[:, 1:]

X_filter = []
for idx, path in enumerate(X_paths):
    raw_path = path[:-11] + "png"
    if raw_path in set(y_paths.values):
        X_filter.append(idx)
X = X_full[X_filter]

print X.shape
print y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
plt.imshow(X_train[0], cmap=plt.cm.gray)

print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

X_train = X_train.reshape(X_train.shape[0], img_x, img_y, 1)
X_test = X_test.reshape(X_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_test = y_test.as_matrix()
y_train = y_train.as_matrix()

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

def on_epoch_end(self, batch, logs={}):
    self.acc.append(logs.get('acc'))


history = AccuracyHistory()

model.fit(X_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_split=0.1,
          callbacks=[history])

plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

y_proba = model.predict_proba(X_test)
y_proba = np.array(y_proba)
threshold = np.arange(0.1, 0.9, 0.1)

acc = []
accuracies = []
best_threshold = np.zeros(y_proba.shape[1])
for i in range(y_proba.shape[1]):
    y_prob = np.array(y_proba[:, i])
    for j in threshold:
        y_pred = [1 if prob >= j else 0 for prob in y_prob]
        acc.append(matthews_corrcoef(y_test[:, i], y_pred))
    acc = np.array(acc)
    index = np.where(acc == acc.max())
    accuracies.append(acc.max())
    best_threshold[i] = threshold[index[0][0]]
    acc = []

print "Class thresholds:", best_threshold
y_pred = np.array([[1 if y_proba[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in
                   range(len(y_test))])

h_loss = hamming_loss(y_test, y_pred)
crisp_accuracy = len([i for i in range(len(y_test)) if (y_test[i] == y_pred[i]).sum() == 5]) / y_test.shape[0]

print "Hamming loss:", h_loss
print "Accuracy:", crisp_accuracy
