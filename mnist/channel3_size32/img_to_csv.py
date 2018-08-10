import numpy as np
import cv2 as cv
import csv

def img_load(path):
    img = cv.imread(path)
    return img

img_list = []
for i in range(10):
    path = 'x_train/%s.png' % str(i)
    img = img_load(path)
    img_list.append(img)

img_array = np.asarray(img_list)
print(img_array.shape)

path = 'x_train.csv'
with open(path, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for i in x_train_array:
        spamwriter.writerow(i)

