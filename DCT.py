import sys
import os
import cv2
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

class LoadImage:
    def __init__(self, file_path):
        self.image = self.read_image(file_path)

    def read_image(self, file_path):
        if not os.path.exists(file_path):
            print("The file path doesn't exist")
            sys.exit()

        try:
            image = cv2.imread(file_path)
            return image

        except Exception as e:
            print("Image open error:", e)
            sys.exit()

class DCTCalculator:
    def __init__(self,image):
        self.high_frequency_coefficients = self.calc_result_hfc(image)

    def rgb2yuv(self,rgb):
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]
        y = 0.299*r + 0.587*g + 0.114*b
        u = -0.14713*r - 0.28886*g + 0.436*b
        v = 0.615*r - 0.51498*g - 0.10001*b
        yuv = np.zeros(rgb.shape)
        yuv[:,:,0] = y
        yuv[:,:,1] = u
        yuv[:,:,2] = v
        return yuv
    
    def calc_result_hfc(self,image):
        rgb_array = np.array(image)
        yuv_array = self.rgb2yuv(rgb_array)
        dct_result = fft.dct(yuv_array)
        high_freq_strength = np.sum(np.abs(dct_result)) - np.sum(np.abs(dct_result[:8, :8]))
        return high_freq_strength
    