import numpy as np

x_train = np.load('x_train.npy')
y_train = np.load('../y_train.npy')
print(x_train.shape)
print(y_train.shape)

x_train_normalize = x_train / 255
print('normalize done.')

from keras.utils import np_utils
y_train_onehot = np_utils.to_categorical(y_train)
print('onehot done.')

from networks.alexnet import AlexNet
model = AlexNet.build()
model.summary()
print('model load.')

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
train_history = model.fit(x_train, y_train_onehot, validation_split = 0.2, epochs = 10, batch_size = 1, verbose = 1,)
model.save('alexnet.h5')
print('model save.')

from keras.models import load_model
model = load_model('alexnet.h5')
scores = model.evaluate(x_test, y_test_onehot, verbose = 0)
print(scores)
