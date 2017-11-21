# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import os


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32)/256
        Y = np.array(Y)
        Y = change(Y)
        return X, Y


def change(target):
    result = np.zeros(shape=[target.shape[0], 10])
    for i in range(0, target.shape[0]):
        value = target[i]
        result[i, value] = 1
    return result


def reader_batch_data(index):
    global file_dir
    path = os.path.join(file_dir, "data_batch_%d" % index)
    print("loading batch file : ", path)
    return load_CIFAR_batch(path)


def reader_test_data():
    global file_dir
    path = os.path.join(file_dir, "test_batch")
    print("loading test file : ", path)
    return load_CIFAR_batch(path)


def init_reader(path):
    global file_dir
    file_dir = path
    global g_x_batch, g_y_batch, g_x_test, g_y_test
    g_x_batch, g_y_batch = reader_batch_data(1)
    g_x_test, g_y_test = reader_test_data()


def get_batch_data(size):
    global batch_index, g_x_batch, g_y_batch, data_index
    if data_index + size > g_x_batch.shape[0]:
        batch_index = batch_index % 5 + 1
        print("batch_index is changed to %d" %batch_index)
        g_x_batch, g_y_batch = reader_batch_data(batch_index)
        data_index = 0
    data_index = data_index + size
    return g_x_batch[data_index:data_index + size], g_y_batch[data_index:data_index + size]


def get_test_data():
    global g_x_test, g_y_test
    return g_x_test, g_y_test


file_dir = None
data_index = 0
batch_index = 1
g_x_batch, g_y_batch, g_x_test, g_y_test = None, None, None, None
