import time
from train.vgg19 import main
from keras import backend as K

def sleep(second):
    for i in range(second):
        print(i)
        time.sleep(1)
        

while True:
    try:
        main('mnist/channel3_32/x_train/', 'style/halftone_32.png', 'mnist/halftone/x_train/')
        print('clear session...')
        K.clear_session()
        sleep(5)
    except:
        print('wait...')        
        sleep(3)
