from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image
import numpy as np


def get_input_data(file_list):
    """
    get input data from files
    :param file_list: files to load
    :return: data
    """
    input_data = []
    for file_name in file_list:
        img = image.load_img(file_name, target_size=(224, 224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0).astype(np.float64)
        x = preprocess_input(x)
        input_data.append(x)
    return input_data


def predict(file_list):
    """
    do prediction of files in file_list
    :param file_list: files
    :return: prediction result
    """
    input_data = get_input_data(file_list)
    vgg19_model = ResNet50()
    predict_rates = vgg19_model.predict(input_data)
    result = decode_predictions(predict_rates,1)
    print(result)


def main():
    file_list = ["./data/2.png"]
    predict(file_list)


if __name__ == "__main__":
    main()
