import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Activation
import os
import PIL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6' 

cifar10 = tf.keras.datasets.cifar10

(pre_x_train, y_train), (pre_x_test, y_test) = cifar10.load_data()
x_train, x_test = pre_x_train / 255, pre_x_test / 255
y_train, y_test = y_train.flatten(), y_test.flatten()

#print("x_train.shape: ", x_train.shape)
#print("y_train.shape: ", y_train.shape)

labels = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
# labels in the CIFAR-10 in the right sequence


K = len(set(y_train))
#print("Number of classes: ", K)


def make_and_train_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test):
    """Makes and trains a model based on the VGG architecture
    this model has 2,4 million parameters."""
    BaseSize = 32

    i = Input(shape=x_train[0].shape)

    x = Conv2D(BaseSize, (3,3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(BaseSize, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(BaseSize*2, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(BaseSize*2, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(BaseSize*4, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(BaseSize*4, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(K, activation='softmax')(x)

    model = Model(i, x)

    print('Model is built...')

    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)

    model.summary()
    model.save('cifar10_model/model.h5')

    return model


# model = make_and_train_model()
# instead of re-making the model everytime we run the script we have it saved and serialized and then we load it instead.

model = tf.keras.models.load_model('cifar10_model/model.h5')


def show_image_and_predict(index):
    """Shows image and the label in the console and shows what the AI guessed and if it was correct.
    takes index in the cifar10 dataset as argument"""
    current_arbritrairy_image = int(index)

    plt.imshow(pre_x_train[current_arbritrairy_image])
    plt.show()


    print("The correct label for this is: ", labels[y_train[current_arbritrairy_image]])

    predictions = model.predict(pre_x_train[current_arbritrairy_image:current_arbritrairy_image+1])[0]
    prediction = list(predictions).index(max(predictions))
    
    print("The AI guessed: ", labels[prediction])
    if labels[prediction] == labels[y_train[current_arbritrairy_image]]:
        print('The AI has guessed the correct label!')
    else:
        print('The AI was incorrect')


#show_image_and_predict(input('>>>'))


def predict_local_image(path_to_image):
    """Attempts to predict what label there is on a local image on the computer.
    takes path to imagefile as argument.
    it rescales the image and saves a temporary file of the rescaled version of the image.
    it normalizes the values and does the calculations and prints the label it predicts.
    """

    img = PIL.Image.open(path_to_image)
    img = img.resize((32,32), PIL.Image.ANTIALIAS)
    img.save('images/temp.jpg')

    array = np.asarray(img)
    array = array / 255
    array.shape = (-1,32,32,3)

    prediction = list(list(model.predict(array))[0])
    prediction = prediction.index(max(prediction))

    print("The AI think it is a:" , labels[prediction])


if __name__ == "__main__":
    predict_local_image('images/plane test img.jpg')