import numpy as np
import cv2 as cv

def img_load(path):
    img = cv.imread(path)
    return img

img_list = []
for i in range(10000):
    path = 'x_test/%s.png' % str(i)
    img = img_load(path)
    img_list.append(img)
    print(path)

img_array = np.asarray(img_list)
print(img_array.shape)

np.save('x_test.npy', img_array)
print('done.')
