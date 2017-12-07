from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from CNN.CIFAR10_recognization.reader import data_set


def train(file_dir):

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    data_set.init_reader(file_dir)
    for i in range(0, 5):
        x_train, y_train = data_set.get_batch_data(10000)
        model.fit(x_train, y_train, epochs=10, batch_size=100)
        if i % 100 == 0 & i > 0:
            print("tesp : ", i)
            model.save("./model/keras_model.h5", overwrite=True, include_optimizer=True)
    x_test, y_test = data_set.get_test_data()
    score = model.evaluate(x_test, y_test, batch_size=50)
    print(score)
    model.save("./model/keras_model.h5", overwrite=True, include_optimizer=True)


def main():
    train("./data")


if __name__ == "__main__":
    main()
