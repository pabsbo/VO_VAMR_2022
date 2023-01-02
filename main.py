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
from Ex8_KLT.track_klt_robustly import trackKLTRobustly

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from state import State
from initialization import initialization
from continuous_operation import processFrame


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

keypoints, landmarks, R_C_W, T_C_W, prev_img = initialization(dataset) 

prev_state = State(keypoints, landmarks)
curr_state = State()

prev_desc = None
prev_kp = keypoints # Keypoints in the first frame
num_frames = len(os.listdir(img_path))

fig = plt.figure()

for i in range(1, num_frames):
    # plt.clf()
    # img = cv2.imread(f'{img_path}' + '/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    # scores = harris(img, corner_patch_size, harris_kappa)
    # kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    # desc = describeKeypoints(img, kp, descriptor_radius)

    # plt.imshow(img, cmap='gray')
    # plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
    # plt.axis('off')

    # if prev_desc is not None:
    #     matches = matchDescriptors(desc, prev_desc, match_lambda)
    #     plotMatches(matches, kp, prev_kp)
    # prev_kp = kp
    # prev_desc = desc

    # plt.pause(0.1)

    ax = fig.add_subplot(2, 1, 1)

    curr_img = cv2.imread(f'{img_path}' + '/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    curr_state, R_C_P, T_C_P, inlier_mask = processFrame(curr_img, prev_img, prev_state, K)

    # Visualize keypoints / their matching / 3D landmarks / Camera movements etc
    ax.imshow(curr_img, cmap='gray')
    ax.plot(prev_state.keypoints[0, :], prev_state.keypoints[1, :], 'bx', linewidth=2)
    ax.plot(curr_state.keypoints[0, :], curr_state.keypoints[1, :], 'rx', linewidth=2)
    ax.axis('off')

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.scatter(curr_state.landmarks[:, 0], curr_state.landmarks[:, 1], curr_state.landmarks[:, 2], s=1)
    drawCamera(ax, -np.matmul(R_C_P.T, T_C_P), R_C_W.T, length_scale=10, head_size=10, set_ax_limits=True)
    print('Frame {} localized with {} inliers!'.format(i, inlier_mask.sum()))

    prev_img = curr_img
    prev_state = curr_state

    plt.pause(0.1)


