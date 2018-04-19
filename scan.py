import argparse
import cv2

import numpy as np
from numpy import genfromtxt
import csv
import operator
from keras.models import load_model

from functions import clean, read_transparent_png

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

model = load_model("model.h5")

image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
if image.shape[2] == 4:
    image = read_transparent_png(args["image"])
image = clean(image)
cv2.imshow('gray', image)
cv2.waitKey(0)

def predict(img):
    image_data = img
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    print(dataset.shape)
    a = model.predict(dataset)[0]

    classes = np.genfromtxt('classes.csv', delimiter=',')[:, 1].astype(int)

    print(classes)
    new = dict(zip(classes, a))
    res = sorted(new.items(), key=operator.itemgetter(1), reverse=True)

    print("#########***#########")
    print("Imagefile = ", args['image'])
    print("Character = ", int(res[0][0]))
    print("Confidence = ", res[0][1] * 100, "%")
    if res[0][1] < 1:
        print("Other predictions")
        for newtemp in res:
            print("Character = ", newtemp[0])
            print("Confidence = ", newtemp[1] * 100, "%")


predict(image)
