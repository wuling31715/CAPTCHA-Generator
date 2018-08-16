from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

class AlexNet:
    
    @staticmethod
    def build(weights_path=None):
        model = Sequential()
        
        #  11x11 conv, 96
        model.add(ZeroPadding2D((1,1), input_shape=(256, 256, 3)))
        model.add(Convolution2D(96, 11, 11, activation='relu'))
        # 5x5 conv, 256
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 5, 5, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        # 3x3 conv, 384
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(384, 3, 3, activation='relu'))
        # 3x3 conv, 384
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(384, 3, 3, activation='relu'))
        # 3x3 conv, 256
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))


        model.add(Flatten())
        # FC-4096
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        # FC-4096
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        # FC-1000
        model.add(Dense(10, activation='softmax'))
        
        return model

    @staticmethod
    def load(weights_path):
        if weights_path:
            model.load_weights(weights_path)        

# if __name__ == "__main__":
#     im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#     im[:,:,0] -= 103.939
#     im[:,:,1] -= 116.779
#     im[:,:,2] -= 123.68
#     im = im.transpose((2,0,1))
#     im = np.expand_dims(im, axis=0)
