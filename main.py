import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Key_Points.harris import harris
from Key_Points.selectKeypoints import selectKeypoints
from Key_Points.describeKeypoints import describeKeypoints
from Key_Points.matchDescriptors import matchDescriptors
from Pose_Estimation.decompose_essential_matrix import decomposeEssentialMatrix
from Pose_Estimation.disambiguate_relative_pose import disambiguateRelativePose
from Pose_Estimation.linear_triangulation import linearTriangulation
from Pose_Estimation.draw_camera import drawCamera
from Pose_Estimation.getMatches import getMatches, pointsToUV, getMatches2, extendPoint

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

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

random_seed = 9

K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
              [0, 7.188560000000e+02, 1.852157000000e+02],
              [0, 0, 1]])

# First and third frame of Kitti dataset
img = cv2.imread('../data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('../data/kitti/05/image_0/000002.png', cv2.IMREAD_GRAYSCALE)

# Select keypoints and descriptors of image 1 and 2
harris_scores = harris(img, corner_patch_size, harris_kappa)
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
descriptors = describeKeypoints(img, keypoints, descriptor_radius)
keypoints = pointsToUV(keypoints)

harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)
keypoints_2 = pointsToUV(keypoints_2)

# Match descriptors between first two images
matches = matchDescriptors(descriptors, descriptors_2, match_lambda)

p1 = np.squeeze(keypoints[[matches[matches >= 0]]])
p2 = np.squeeze(keypoints_2[[matches[matches >= 0]]])

rng = np.random.default_rng(random_seed)

# Apply Ransac to Obtain the Essential Matrix and inliers
model, inliers = ransac((p1, p2), FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000,
                        random_state=rng)

inliers = inliers.astype(int)
inlier_keypoints_1 = np.squeeze(p1[[inliers[inliers > 0]]])
inlier_keypoints_2 = np.squeeze(p2[[inliers[inliers > 0]]])

F = model.params

E = np.linalg.inv(K).T @ F @ np.linalg.inv(K)

Rots, u3 = decomposeEssentialMatrix(E)

R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, extendPoint(inlier_keypoints_1), extendPoint(inlier_keypoints_2), K, K)

M1 = K @ np.eye(3, 4)
M2 = K @ np.c_[R_C2_W, T_C2_W]
P = linearTriangulation(extendPoint(inlier_keypoints_1), extendPoint(inlier_keypoints_2), M1, M2)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.scatter(P[0, :], P[1, :], P[2, :], marker='o')

# Display camera pose
drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale=2)
ax.text(-0.1, -0.1, -0.1, "Cam 1")

center_cam2_W = -R_C2_W.T @ T_C2_W
drawCamera(ax, center_cam2_W, R_C2_W.T, length_scale=2)
ax.text(center_cam2_W[0] - 0.1, center_cam2_W[1] - 0.1, center_cam2_W[2] - 0.1, 'Cam 2')

# Display matched points
ax = fig.add_subplot(1, 3, 2)
ax.imshow(img)
ax.scatter(p1[0, :], p1[1, :], color='y', marker='s')
ax.set_title("Image 1")

ax = fig.add_subplot(1, 3, 3)
ax.imshow(img_2)
ax.scatter(p2[0, :], p2[1, :], color='y', marker='s')
ax.set_title("Image 2")

plt.show()
