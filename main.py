import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from Ex3_harris.harris import harris
from Ex3_harris.selectKeypoints import selectKeypoints
from Ex3_harris.describeKeypoints import describeKeypoints
from Ex3_harris.matchDescriptors import matchDescriptors
from Ex3_harris.plotMatches import plotMatches
from Ex6_triangulation.decompose_essential_matrix import decomposeEssentialMatrix
from Ex6_triangulation.disambiguate_relative_pose import disambiguateRelativePose
from Ex6_triangulation.linear_triangulation import linearTriangulation
from Ex6_triangulation.draw_camera import drawCamera
from Ex6_triangulation.estimate_essential_matrix import estimateEssentialMatrix
from Ex7_ransac.ransacFundamentalMatrix import ransacFundamentalMatrix

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from initialization import initialization

matplotlib.use('TkAgg')

# Parameters used in previous exercises
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200 #1000
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 5

baseline = 0.54
patch_radius = 5
min_disp = 5
max_disp = 50

random_seed = 9
dataset = 'KITTI' # 'KITTI', 'PARKING'
if dataset == 'KITTI':
    img_path = '../data/kitti/05/image_0'
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                  [0, 7.188560000000e+02, 1.852157000000e+02],
                  [0, 0, 1]])

elif dataset == 'PARKING':
    img_path = '../data/parking/images'
    K = np.array([[331.37, 0, 320],
                  [0, 369.568, 240],
                  [0, 0, 1]])

p, P, R_C_W, T_C_W = initialization(img_path) # p: keypoints in frame 0, P: landmarks, R_C_W: camera rotation matrix, T_C_W: camera position







# Part 5 - Match descriptors between all images
prev_desc = None
prev_kp = None
num_frames = len(os.listdir(img_path))
for i in range(num_frames):
    plt.clf()
    img = cv2.imread(f'{img_path}' + '/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    scores = harris(img, corner_patch_size, harris_kappa)
    kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    desc = describeKeypoints(img, kp, descriptor_radius)

    plt.imshow(img, cmap='gray')
    plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
    plt.axis('off')

    if prev_desc is not None:
        matches = matchDescriptors(desc, prev_desc, match_lambda)
        plotMatches(matches, kp, prev_kp)
    prev_kp = kp
    prev_desc = desc

    plt.pause(0.1)