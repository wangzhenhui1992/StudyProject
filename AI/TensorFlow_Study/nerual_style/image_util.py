# -*- utf-8 -*-
"""
utils to deal with images
"""
import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from PIL import Image
import os


def load_img(path):
    """
    this method is used to read image data from previous path for vgg19
    :param path: image file
    :return: np.array object with shape(1,224,224,3) , ordered by RGB->BGR ,centering zero
    """
    img = image.load_img(path, target_size=(224, 224))
    arr = np.expand_dims(image.img_to_array(img), axis=0)
    arr = preprocess_input(arr)
    return arr


def save_image(image_data, dir_name, image_name):
    """
    change the image_data and save it as an image
    :param image_data: image data
    :param dir_name: path of dir
    :param image_name: file name
    :return: None
    """
    image_data = image_data.astype(dtype=np.float64)
    image_arr = np.reshape(image_data, [224, 224, 3])
    image_arr[..., 0] += 103.939
    image_arr[..., 1] += 116.779
    image_arr[..., 2] += 123.68
    image_arr = image_arr[..., ::-1]
    image_arr = np.clip(image_arr, 0, 255).astype(dtype=np.uint8)
    full_path = os.path.join(dir_name, image_name)
    Image.fromarray(image_arr).save(full_path)
