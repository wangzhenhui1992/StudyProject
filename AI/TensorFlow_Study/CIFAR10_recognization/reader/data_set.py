# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import os

file_dir = None
data_index = 0
batch_index = 1
g_x_batch, g_y_batch, g_x_test, g_y_test = None, None, None, None


def load_CIFAR_batch(file_name):
    """
    load cifar10 data from file
    :param file_name:
    :return:
    """
    with open(file_name, 'rb')as f:
        datadict = p.load(f, encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        print(x.shape)
        x = x.reshape(10000, 3, 32, 32) / 256
        x = np.transpose(x,axes=[0,2,3,1])
        y = np.array(y)
        y = change(y)
        return x, y


def change(target):
    """
    do data format changing
    :param target:
    :return:
    """
    result = np.zeros(shape=[target.shape[0], 10])
    for i in range(0, target.shape[0]):
        value = target[i]
        result[i, value] = 1
    return result


def batch_data(index):
    """
    provide data with batch index
    :param index:
    :return:
    """
    global file_dir
    path = os.path.join(file_dir, "data_batch_%d" % index)
    print("loading batch file : ", path)
    return load_CIFAR_batch(path)


def test_data():
    global file_dir
    path = os.path.join(file_dir, "test_batch")
    print("loading test file : ", path)
    return load_CIFAR_batch(path)


def init_reader(path):
    global file_dir
    file_dir = path
    global g_x_batch, g_y_batch, g_x_test, g_y_test
    g_x_batch, g_y_batch = batch_data(1)
    g_x_test, g_y_test = test_data()


def get_batch_data(size):
    global batch_index, g_x_batch, g_y_batch, data_index
    print(data_index," ",size)
    if data_index + size > g_x_batch.shape[0]:
        batch_index = batch_index % 5 + 1
        print("batch_index is changed to %d" % batch_index)
        g_x_batch, g_y_batch = batch_data(batch_index)
        data_index = 0
    x , y = g_x_batch[data_index:data_index + size], g_y_batch[data_index:data_index + size]
    data_index += size
    return x,y


def get_test_data():
    global g_x_test, g_y_test
    return g_x_test, g_y_test


def main():
    load_CIFAR_batch("../data/data_batch_1")


if __name__ == "__main__":
    main()
