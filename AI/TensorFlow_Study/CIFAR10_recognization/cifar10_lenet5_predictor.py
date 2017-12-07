from keras.models import load_model
from PIL import Image
import numpy as np
from CNN.CIFAR10_recognization.reader import data_set

model = load_model("./model/keras_model.h5", compile=True)

data_set.init_reader('./data')
x_test, y_test = data_set.get_batch_data(8000)
print(x_test.shape)
score = model.evaluate(x_test, y_test, batch_size=50)
print(score)

im = Image.open(r'./data/dog2.png')
img = np.array(im)
print(img.shape)
if img.shape[2] > 3:
    img = img[0:img.shape[0], 0:img.shape[1], 0:3]
input_image = img.reshape((1, 32, 32, 3)) / 256

print(np.argmax(model.predict(x=input_image, batch_size=1)))
