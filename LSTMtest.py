import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics


def create_sequence(dataset, n):
    sequence = []
    for i in range(len(dataset)):
        if i % n == 0:
            sequence.append(dataset[i: i + 25])

    return sequence


def one_hot_encoder(data):
    one_hot_y = tf.one_hot(data.astype(np.int32), depth=5, axis=1)
    return one_hot_y


parser = argparse.ArgumentParser(description='Training lstm model')
parser.add_argument('--X_test_path', help='Path to testing csv file.')
parser.add_argument('--Y_test_path', help='Path to testing csv file.')
parser.add_argument('--sequence_length', help='length of each sequence')
parser.add_argument('--model',help='lstm model to test')
args = parser.parse_args()

X_test_path = args.X_test_path
Y_test_path = args.Y_test_path

X_test_csv = pd.read_csv(X_test_path, header=None, encoding='utf-7')
Y_test_csv = pd.read_csv(Y_test_path, header=None, encoding='utf-7')

test_dataset = X_test_csv.values

X_test_dataset = test_dataset.astype(float)
Y_test_dataset = Y_test_csv.values.tolist()

X_test_list = create_sequence(X_test_dataset, args.sequence_length)
Y_test = np.array(one_hot_encoder(Y_test_dataset)).reshape(-1, 5)

X_test = np.array(X_test_list)

lstm = tf.keras.models.load_model(args.model)
pred = lstm.predict(X_test)
n_classes = 5
LABELS = [
    "SIT",
    "WALK",
    "BEND",
    "STAND",
    "FALL"
    ]
precision, recall, f_score, support = metrics.precision_recall_fscore_support(Y_test_dataset, pred.argmax(1),average="weighted")

print("precision",100 * precision)
print("recall", 100 * recall)
print("f_score",100 * f_score)
print("number of occurrences of each class in test data", support)

print("confusion matrix")
confusion_matrix_basic = metrics.confusion_matrix(Y_test_dataset, pred.argmax(1))
print(confusion_matrix_basic)

cm = confusion_matrix(Y_test.argmax(1), pred.argmax(1))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, format(cm[i, j], fmt),
  horizontalalignment="center",
  color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


