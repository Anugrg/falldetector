import pandas as pd
import argparse
import os
import os.path
import numpy as np
import re

""" This program assumes that the csv files have 
been divided according to the camera channels or angles and the action
classes 
"""


def process_csv(csv, label):
    # global X_train
    temp = csv.values
    data = temp.astype(float)
    x = []
    y = []
    if len(data) < 40:
        print("less than 40 frames")
        pad = create_pad()
        print("pad",pad)
        data = pad_sequence(data, pad)
        print("data",data)
        x.append(data)
        print(np.array(x).shape)
    elif len(data) >= 40:
        print("greater than 40 frames")
        for i in range(40, len(data)):
            x.append(data[i - 40:i])
    n = len(x)
    print("n",n)
    y = create_labels(n, label)
    return x, y


def create_labels(n, len):
    # global Y_train
    Y = []
    if len == 0:
        for j in range(n):
            Y_train.append([0])
    elif len == 1:
        for j in range(n):
            Y_train.append([1])
    elif len == 2:
        for j in range(n):
            Y_train.append([2])
    elif len == 3:
        for j in range(n):
            Y_train.append([3])
    elif len == 4:
        for j in range(n):
            Y_train.append([4])
    return Y


def pad_sequence(data, pad):
    print("inside pad sequence")
    print(data)
    print("pad",pad)
    if len(data) < 40:
        gap = 40 - len(data)
        print(gap)
        for i in range(gap):
            data = np.vstack((data, pad))
            # data.append(pad)
    return data


def create_pad():
    pad = []
    for i in range(0, 36):
        pad.append(0.0)
    return pad


parser = argparse.ArgumentParser(description='creating training file')
parser.add_argument('--directory', help='Path to directory containing all csv files.')
args = parser.parse_args()

# ch1, ch2, ch3, ch4
# fall, sit, walk, bend, stand

X_train = []
Y_train = []
label = 0

for dir_path, dir_names, files in os.walk(args.directory):
    for file in files:
        if re.search("sit", file):
            label = 0
        elif re.search("stand", file):
            label = 1
        elif re.search("walk", file):
            label = 2
        elif re.search("bend", file):
            label = 3
        elif re.search("fall", file):
            label = 4

        path = dir_path + "/" + file
        print(path)
        csv = pd.read_csv(path, header=None, encoding='utf-7')
        # csv = csv.drop(csv.index[0])
        print (csv)
        for i in range(36,39):
            csv.drop([i], axis=1, inplace=True)
        print(csv)
        temp_x, temp_y = process_csv(csv, label)
        X_train.extend(temp_x)
        Y_train.extend(temp_y)


print(np.array(X_train).shape)
print(np.array(Y_train).shape)

print(X_train[0][34].shape)

print(np.array(Y_train))

