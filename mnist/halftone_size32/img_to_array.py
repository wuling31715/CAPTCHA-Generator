import numpy as np
import cv2 as cv
import csv
import os

def img_load(path):
    img = cv.imread(path)
    return img

img_list = []
path = 'x_test/'
n = 0
for i in os.listdir(path):
    if '.png' in i:
        n += 1

for i in range(n):
    path = 'x_test/%s.png' % str(i)
    img = img_load(path)
    img_list.append(img)
    print(path)

img_array = np.asarray(img_list)
print(img_array.shape)

np.save('x_test.npy', img_array)
test = np.load('x_test.npy')
print(test.shape)

if img_array.shape == test.shape:
    print('done')
