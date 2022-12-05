import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from VO_VAMR_2022.Key_Points.shi_tomasi import shi_tomasi
from VO_VAMR_2022.Key_Points.harris import harris
from VO_VAMR_2022.Key_Points.selectKeypoints import selectKeypoints
from VO_VAMR_2022.Key_Points.describeKeypoints import describeKeypoints
from VO_VAMR_2022.Key_Points.matchDescriptors import matchDescriptors
from VO_VAMR_2022.Key_Points.plotMatches import plotMatches
from VO_VAMR_2022.Pose_Estimation.estimate_essential_matrix import estimateEssentialMatrix
from VO_VAMR_2022.Pose_Estimation.decompose_essential_matrix import decomposeEssentialMatrix
from VO_VAMR_2022.Pose_Estimation.disambiguate_relative_pose import disambiguateRelativePose
from VO_VAMR_2022.Pose_Estimation.linear_triangulation import linearTriangulation
from VO_VAMR_2022.Pose_Estimation.draw_camera import drawCamera

matplotlib.use('TkAgg')
# Randomly chosen parameters that seem to work well - can you find better ones?
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 4

img = cv2.imread('../data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)

# Part 1 - Calculate Corner Response Functions
# Shi-Tomasi
shi_tomasi_scores = shi_tomasi(img, corner_patch_size)

# Harris
harris_scores = harris(img, corner_patch_size, harris_kappa)
#
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].axis('off')
axs[0, 1].imshow(img, cmap='gray')
axs[0, 1].axis('off')

axs[1, 0].imshow(shi_tomasi_scores)
axs[1, 0].set_title('Shi-Tomasi Scores')
axs[1, 0].axis('off')

axs[1, 1].imshow(harris_scores)
axs[1, 1].set_title('Harris Scores')
axs[1, 1].axis('off')

fig.tight_layout()
plt.show()

# Part 2 - Select keypoints
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)

plt.clf()
plt.close()
plt.imshow(img, cmap='gray')
plt.plot(keypoints[1, :], keypoints[0, :], 'rx', linewidth=2)
plt.axis('off')
plt.show()

# Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
descriptors = describeKeypoints(img, keypoints, descriptor_radius)

plt.clf()
plt.close()
fig, axs = plt.subplots(4, 4)
patch_size = 2 * descriptor_radius + 1
for i in range(16):
    axs[i // 4, i % 4].imshow(descriptors[:, i].reshape([patch_size, patch_size]))
    axs[i // 4, i % 4].axis('off')

plt.show()

# Part 4 - Match descriptors between first two images
img_2 = cv2.imread('../data/kitti/05/image_1/000000.png', cv2.IMREAD_GRAYSCALE)
harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)

matches = matchDescriptors(descriptors_2, descriptors, match_lambda)
plt.clf()
plt.close()
plt.imshow(img_2, cmap='gray')
plt.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
plotMatches(matches, keypoints_2, keypoints)
plt.tight_layout()
plt.axis('off')
plt.show()

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
