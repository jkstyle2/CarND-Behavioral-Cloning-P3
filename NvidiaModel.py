from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
# from keras.layers.normalization import BatchNormalization

def NvidiaModel():
    '''
    Nvidia model is used because of its simplicity and demonstrated ability
    to perform well on self-driving car tasks.
    '''

    # row, col, ch = 160, 320, 3  # original image size
    row, col, ch = 70, 160, 3
    activation_type = 'relu'
    dropout_prob = 0.3

    model = Sequential()

    # Normalization Layer
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))

    # Convolutional Layer 1 : input: (70,160,3) , output: (33,78,24)
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation(activation_type))
    model.add(Dropout(dropout_prob))

    # Convolutional Layer 2 : input: (33,78,24) , output: (15,37,36)
    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation(activation_type))
    model.add(Dropout(dropout_prob))

    # Convolutional Layer 3 : input: (15,37,36) , output: (6,17,48)
    model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation(activation_type))
    model.add(Dropout(dropout_prob))

    # Convolutional Layer 4 : input: (6,17,48) , output: (5,16,64)
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
    model.add(Activation(activation_type))
    model.add(Dropout(dropout_prob))

    # Convolutional Layer 5 : input: (5,16,64) , output: (4,15,64)
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
    model.add(Activation(activation_type))
    model.add(Dropout(dropout_prob))

    # Flatten Layer : input: (4,15,64) , output: (1,3840)
    model.add(Flatten())

    # Fully-connected Layer 1 : input: (1,3840) , output: (1,100)
    model.add(Dense(100))
    model.add(Activation(activation_type))

    # Fully-connected Layer 2 : input: (1,100) , output: (1,50)
    model.add(Dense(50))
    model.add(Activation(activation_type))

    # Fully-connected Layer 3 : input: (1,50) , output: (1,10)
    model.add(Dense(10))
    model.add(Activation(activation_type))

    # Output Layer : input: (1,10) , output: (1,1)
    model.add(Dense(1))

    return model