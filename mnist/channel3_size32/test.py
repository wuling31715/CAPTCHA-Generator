import numpy as np

x_test = np.load('x_test.npy')
y_test = np.load('../y_test.npy')
print(x_test.shape)
print(y_test.shape)

x_train_normalize = x_train / 255
print('normalize done.')


from keras.utils import np_utils
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
print('onehot done.')

from keras.models import load_model
model = load_model('models/alexnet.h5')
print('model load.')

scores = model.evaluate(x_train_normalize, y_test_onehot, verbose = 0)
print(scores)