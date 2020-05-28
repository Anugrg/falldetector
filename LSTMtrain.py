import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import itertools


def create_sequence(dataset, n):
    sequence = []
    for i in range(len(dataset)):
        if i % n == 0:
            sequence.append(dataset[i: i + 25])

    return sequence


def build_model(time_steps, n_features, units):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(units, return_sequences=True, input_shape=(time_steps, n_features)),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    return model


def one_hot_encoder(data):
    one_hot_y = tf.one_hot(data.astype(np.int32), depth=5, axis=1)
    return one_hot_y


parser = argparse.ArgumentParser(description='Training lstm model')
parser.add_argument('--X_train_path', help='Path to training csv file.')
parser.add_argument('--Y_train_path', help='Path to training csv file.')
parser.add_argument('--sequence_length', help='length of each sequence')
parser.add_argument('--hidden_units', help='hidden units of lstm')
parser.add_argument('--batch_size', help='batch size for training')
parser.add_argument('--epochs', help='number of epochs to train')
args = parser.parse_args()

X_train_path = args.X_train_path
Y_train_path = args.Y_train_path

X_train_csv = pd.read_csv(X_train_path, header=None, encoding='utf-7')
Y_train_csv = pd.read_csv(Y_train_path, header=None, encoding='utf-7')

train_dataset = X_train_csv.values
X_train_dataset = train_dataset.astype(float)
Y_train_dataset = Y_train_csv.values.tolist()

X_train_list = create_sequence(X_train_dataset, args.sequence_length)
Y_train = np.array(one_hot_encoder(Y_train_dataset)).reshape(-1, 5)

X_train = np.array(X_train_list)
print("x train:", X_train.shape)

# for imbalanced class otherwise comment
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), np.array(Y_train_dataset).reshape(-1))
class_weight = {0: class_weight[0], 1: class_weight[1], 2: class_weight[2], 3: class_weight[3], 4: class_weight[4]}

# Below, we define the keras lstm layers
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

LSTM_model = build_model(args.sequence_length, 36, args.hidden_units)

LSTM_model.summary()

LSTM_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025),
                   loss='categorical_crossentropy',
                   metrics=[tf.keras.metrics.CategoricalCrossentropy(),
                            'accuracy'])

history = LSTM_model.fit(X_train, Y_train, batch_size=args.batch_size,
                         validation_split=0.2, callbacks=[callback],
                         epochs=args.epochs, class_weight=class_weight)  # remove arguments which are not not used

plt.plot(history.epoch, history.history['loss'], linestyle="--")
plt.plot(history.epoch, history.history['accuracy'] )

plt.title('Model training results')
plt.ylabel('loss or accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_loss', 'Train_accuracy'], loc='upper left')
plt.show()

plt.plot(history.epoch, history.history['val_loss'], linestyle="--")
plt.plot(history.epoch, history.history['val_accuracy'] )

plt.title('Model training results')
plt.ylabel('loss or accuracy ')
plt.xlabel('Epoch')
plt.legend(['val_loss', 'val_accuracy'], loc='upper left')
plt.show()

plt.plot(history.epoch, history.history['categorical_crossentropy'])
plt.title('Model categorical crossentropy')
plt.ylabel('progress')
plt.xlabel('Epoch')
plt.legend(['categorical_crossentropy'],loc = 'upper left')
plt.show()

keras.utils.plot_model(LSTM_model, 'multi_input_and_output_model.png', show_shapes=True)

#LSTM_model.save_weights('path/to/folder')
LSTM_model.save('lstm.h5' )

