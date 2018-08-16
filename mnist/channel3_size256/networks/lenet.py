from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class LeNet:
    @staticmethod
    def build():
        model = Sequential()
        #Layer 1
        #Conv Layer 1
        model.add(Conv2D(input_shape = (256, 256, 3), filters = 6, kernel_size = 5, strides = 1, activation = 'relu'))
        #Pooling layer 1
        model.add(MaxPooling2D(pool_size = 2, strides = 2))
        #Layer 2
        #Conv Layer 2
        model.add(Conv2D(input_shape = (14, 14, 6), filters = 16,  kernel_size = 5, strides = 1, activation = 'relu'))
        #Pooling Layer 2
        model.add(MaxPooling2D(pool_size = 2, strides = 2))
        #Flatten
        model.add(Flatten())
        #Layer 3
        #Fully connected layer 1
        model.add(Dense(units = 120, activation = 'relu'))
        #Layer 4
        #Fully connected layer 2
        model.add(Dense(units = 84, activation = 'relu'))
        #Layer 5
        #Output Layer
        model.add(Dense(units = 10, activation = 'softmax'))
        print('model build.')
        return model
