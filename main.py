import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from collections import deque

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

dataset = 'PARKING' # 'KITTI', 'PARKING'
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

num_frames = len(os.listdir(img_path))

num_tracked_landmarks = deque(maxlen=20)
for i in range(num_tracked_landmarks.maxlen):
    num_tracked_landmarks.append(0)

full_trajectory_x = deque()
full_trajectory_y = deque()

fig = plt.figure(figsize=(12,6))
gs=GridSpec(2,4)

for i in range(1, num_frames):
    fig.clear()
    if dataset == 'KITTI':
        curr_img = cv2.imread(f'{img_path}' + '/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    elif dataset == 'PARKING':
        curr_img = cv2.imread(f'{img_path}' + '/img_{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)

    curr_state, R_C_P, T_C_P, inlier_mask = processFrame(curr_img, prev_img, prev_state, K)

    # Visualize current image with keypoints / their matching
    ax = fig.add_subplot(gs[0,:2])
    ax.imshow(curr_img, cmap='gray')
    ax.plot(prev_state.keypoints[0, :], prev_state.keypoints[1, :], 'bx', linewidth=2)
    ax.plot(curr_state.keypoints[0, :], curr_state.keypoints[1, :], 'rx', linewidth=2)
    # ax.plot(curr_state.candidate_keypoints[0, :], curr_state.candidate_keypoints[1, :], 'gx', linewidth=2)
    ax.set_title(f'Current Image, Frame # {i}')
    ax.axis('off')

    # Visualize # tracked landmarks over last 20 frames
    ax = fig.add_subplot(gs[1,0])
    num_tracked_landmarks.append(curr_state.landmarks.shape[1])
    ax.plot(np.flip(-np.arange(20)), np.array(num_tracked_landmarks))
    ax.set_title('# tracked landmarks over last 20 frames')

    # Visualize Full Trajectory
    ax = fig.add_subplot(gs[1,1])
    camera_position = -np.matmul(R_C_P.T, T_C_P) # TODO: This might be incorrect. Should I multiply with the previous rot.matrix?
    full_trajectory_x.append(camera_position[0])
    full_trajectory_y.append(camera_position[1])
    ax.plot(np.array(full_trajectory_x), np.array(full_trajectory_y), color='r')
    ax.set_title('Full Trajectory')

    # Visualize Trajectory of last 20 frames and landmarks
    ax = fig.add_subplot(gs[:,2:])
    # ax = fig.add_subplot(gs[0,2])
    ax.plot(np.arange(2), np.arange(2))
    ax.scatter(curr_state.landmarks[0,:], curr_state.landmarks[1,:], color='k', s=5)
    ax.set_title('Trajectory of last 20 frames and landmarks')


    # Visualize 3D landmarks / Camera pose (debug)
    # ax = fig.add_subplot(gs[0,3], projection='3d')
    # ax.set_xlim3d(-1, 5)
    # ax.set_ylim3d(-1, 5)
    # ax.set_zlim3d(-1, 5)
    # ax.scatter(curr_state.landmarks[:, 0], curr_state.landmarks[:, 1], curr_state.landmarks[:, 2], s=1)
    # drawCamera(ax, -np.matmul(R_C_P.T, T_C_P), R_C_P.T, length_scale=10, head_size=10, set_ax_limits=True)
    # ax.set_title('3D Landmarks and Camera Pose')
    # print('Frame {} localized with {} inliers!'.format(i, inlier_mask.sum()))

    prev_img = curr_img
    prev_state = curr_state

    plt.pause(0.1)


