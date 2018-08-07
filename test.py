import time
from train.vgg19 import main

while True:
    main('mnist/channel3_32/x_train/', 'style/halftone_32.png', 'mnist/halftone/x_train/')    
    time.sleep(30)
    