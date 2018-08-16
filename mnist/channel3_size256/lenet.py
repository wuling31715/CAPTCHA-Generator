import numpy as np

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('../y_train.npy')
y_test = np.load('../y_test.npy')
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train_normalize = x_train / 255
x_test_normalize = x_test / 255

from keras.utils import np_utils
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

from networks.lenet import LeNet
model = LeNet.build()
model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
train_history = model.fit(x_train_normalize, y_train_onehot, validation_split = 0.2, epochs = 10, batch_size = 128, verbose = 1,)
model.save('lenet.h5')
print('model save.')

from keras.models import load_model
model = load_model('lenet.h5')
scores = model.evaluate(x_test_normalize, y_test_onehot, verbose = 0)
print(scores)
