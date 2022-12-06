import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Key_Points.shi_tomasi import shi_tomasi
from Key_Points.harris import harris
from Key_Points.selectKeypoints import selectKeypoints
from Key_Points.describeKeypoints import describeKeypoints
from Key_Points.matchDescriptors import matchDescriptors
from Pose_Estimation.plotMatches import plotMatches
from Pose_Estimation.disparityToPointCloud import disparityToPointCloud
from Pose_Estimation.estimate_essential_matrix import estimateEssentialMatrix
from Pose_Estimation.decompose_essential_matrix import decomposeEssentialMatrix
from Pose_Estimation.disambiguate_relative_pose import disambiguateRelativePose
from Pose_Estimation.getDisparity import getDisparity
from Pose_Estimation.linear_triangulation import linearTriangulation
from Pose_Estimation.draw_camera import drawCamera
from Pose_Estimation.reprojectPoints import reprojectPoints
from Pose_Estimation.ransacLocalization import ransacLocalization

matplotlib.use('TkAgg')

# Parameters used in previous exercises
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 1000
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 5

baseline = 0.54
patch_radius = 5
min_disp = 5
max_disp = 50

K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
              [0, 7.188560000000e+02, 1.852157000000e+02],
              [0, 0, 1]])

# First and third frame of Kitti dataset
img = cv2.imread('../data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('../data/kitti/05/image_1/000000.png', cv2.IMREAD_GRAYSCALE)

# Select keypoints and descriptors of image 1 and 2
harris_scores = harris(img, corner_patch_size, harris_kappa)
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
descriptors = describeKeypoints(img, keypoints, descriptor_radius)

harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)

# Match descriptors between first two images
matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

# First plot of matching keypoints with outliers
# plt.clf()
# plt.close()
# plt.imshow(img_2, cmap='gray')
# plt.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
# plotMatches(matches, keypoints_2, keypoints)
# plt.tight_layout()
# plt.axis('off')
# plt.show()

# We have to find the 3D coordinates of the matches, I'm trying to use the ex5 point cloud part to do it.
# Get Disparity of the two images
disp_img = getDisparity(img, img_2, patch_radius, min_disp, max_disp)
# Create point cloud for first pair
p_C_points, intensities = disparityToPointCloud(disp_img, K, baseline, img)
T_C_F = np.asarray([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
p_F_points = np.matmul(np.linalg.inv(T_C_F), p_C_points[::11, :, None]).squeeze(-1)

# Drop unmatched keypoints and get 3d landmarks for the matched ones.
matched_query_keypoints = keypoints_2[:, matches >= 0]
corresponding_matches = matches[matches >= 0]
corresponding_landmarks = p_F_points[corresponding_matches, :]

# perform RANSAC to find best Pose and inliers
# TODO the probles is here, I don't know why the func.ransacLocalization is not working with the values that we're passing
out = ransacLocalization(matched_query_keypoints, corresponding_landmarks, K)
R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = out


# Ignore the next code lines

# Part 5 - Match descriptors between all images
# prev_desc = None
# prev_kp = None
# for i in range(200):
#     plt.clf()
#     img = cv2.imread('../data/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
#     scores = harris(img, corner_patch_size, harris_kappa)
#     kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
#     desc = describeKeypoints(img, kp, descriptor_radius)
#
#     plt.imshow(img, cmap='gray')
#     plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
#     plt.axis('off')
#
#     if prev_desc is not None:
#         matches = matchDescriptors(desc, prev_desc, match_lambda)
#         plotMatches(matches, kp, prev_kp)
#     prev_kp = kp
#     prev_desc = desc
#
#     plt.pause(0.1)

#
# K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
#               [0, 7.188560000000e+02, 1.852157000000e+02],
#               [0, 0, 1]])
# # Load outlier-free point correspondences
# p1 = np.loadtxt('../data/matches0001.txt')
# p2 = np.loadtxt('../data/matches0002.txt')
#
# p1 = np.r_[p1, np.ones((1, p1.shape[1]))]
# p2 = np.r_[p2, np.ones((1, p2.shape[1]))]
#
# # Estimate the essential matrix E using the 8-point algorithm
# E = estimateEssentialMatrix(p1, p2, K, K)
# print("E:\n", E)
#
# # Extract the relative camera positions (R,T) from the essential matrix
# # Obtain extrinsic parameters (R,t) from E
# Rots, u3 = decomposeEssentialMatrix(E)
#
# # Disambiguate among the four possible configurations
# R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)
#
# # Triangulate a point cloud using the final transformation (R,T)
# M1 = K @ np.eye(3, 4)
# M2 = K @ np.c_[R_C2_W, T_C2_W]
# P = linearTriangulation(p1, p2, M1, M2)
#
# # Visualize the 3-D scene
# fig = plt.figure()
# ax = fig.add_subplot(1, 3, 1, projection='3d')
#
# # R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]
#
# # P is a [4xN] matrix containing the triangulated point cloud (in
# # homogeneous coordinates), given by the function linearTriangulation
# ax.scatter(P[0, :], P[1, :], P[2, :], marker='o')
#
# # Display camera pose
# drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale=2)
# ax.text(-0.1, -0.1, -0.1, "Cam 1")
#
# center_cam2_W = -R_C2_W.T @ T_C2_W
# drawCamera(ax, center_cam2_W, R_C2_W.T, length_scale=2)
# ax.text(center_cam2_W[0] - 0.1, center_cam2_W[1] - 0.1, center_cam2_W[2] - 0.1, 'Cam 2')
#
# # Display matched points
# ax = fig.add_subplot(1, 3, 2)
# ax.imshow(img)
# ax.scatter(p1[0, :], p1[1, :], color='y', marker='s')
# ax.set_title("Image 1")
#
# ax = fig.add_subplot(1, 3, 3)
# ax.imshow(img_2)
# ax.scatter(p2[0, :], p2[1, :], color='y', marker='s')
# ax.set_title("Image 2")
#
# plt.show()
