import tensorflow as tf
import numpy as np
from PIL import Image

im = Image.open("./image/50.png")
img = np.array(im)
print(img[img < 0])



def load_img(img_path):
    im = Image.open(img_path)
    img = np.array(im)
    if img.shape[2] > 3:
        img = img[:, :, 0:3]
    img = np.expand_dims(img, axis=0)
    return img


def get_train_image():
    im = Image.open(r"./data/dog.png")
    im.resize((100, 100))
    img = np.array(im)
    if img.shape[2] > 3:
        img = img[:, :, 0:3]
    img = np.expand_dims(img, axis=0)
    return img


