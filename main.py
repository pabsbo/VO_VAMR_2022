import numpy as np
import math
from matplotlib import image as mpimg
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Select the image number 1 and three of the KITTI dataset

left_img = cv2.resize(cv2.imread('data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
right_img = cv2.resize(cv2.imread('data/kitti/05/image_1/000000.png', cv2.IMREAD_GRAYSCALE), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
