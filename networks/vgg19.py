from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

class VGG19:
    
    @staticmethod
    def build(weights_path=None):
        model = Sequential()

        # conv3-64
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        # conv3-64
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        # conv3-128
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        # conv3-128
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        # conv3-256
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        # conv3-256
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        # conv3-256
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        # conv3-256
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # conv3-512
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        # maxpool
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        # FC-4096
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # FC-4096
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # FC-1000
        model.add(Dense(1000, activation='softmax'))

        return model

    @staticmethod
    def load(weights_path):
        if weights_path:
            model.load_weights(weights_path)        

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_19('vgg19_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
print np.argmax(out)