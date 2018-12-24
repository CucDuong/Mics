import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time

def load_image(path):
    img = cv2.imread(path)
    plt.imshow(img)
    return img

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])
