import time
from train.vgg19 import main
from keras import backend as K


while True:
    main('mnist/channel3_32/x_train/', 'style/halftone_32.png', 'mnist/halftone/x_train/')    
    K.clear_session()
    time.sleep(5)
    