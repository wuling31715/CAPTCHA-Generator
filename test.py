import time
from train.vgg19 import main
from keras import backend as K

K.clean_session()
while True:
    try:
        main('mnist/channel3_32/x_train/', 'style/halftone_32.png', 'mnist/halftone/x_train/')
    except:
        print('wait')
        time.sleep(3)
