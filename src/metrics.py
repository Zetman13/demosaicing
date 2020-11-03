import cv2
import numpy as np


def mse(x1, x2):
    y1 = cv2.cvtColor(x1.astype('uint8'), cv2.COLOR_RGB2GRAY)
    y2 = cv2.cvtColor(x2.astype('uint8'), cv2.COLOR_RGB2GRAY)
    return np.square(y1-y2).mean()


def psnr(x1, x2):
    return 10*np.log10(255**2/mse(x1, x2))
