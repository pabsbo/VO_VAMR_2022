import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
from Ex6_triangulation.normalise_2D_pts import normalise2DPts
from Ex7_ransac.ransacFundamentalMatrix import ransacFundamentalMatrix
from Ex7_ransac.ransacLocalization import ransacLocalization
from Ex8_KLT.track_klt_robustly import trackKLTRobustly

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

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

# KLT parameters
r_T = 15
n_iter = 50
threshold = 0.1

DATASET = 'KITTI' # 'KITTI', 'PARKING'

if DATASET == 'KITTI':
    # KITTI
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                  [0, 7.188560000000e+02, 1.852157000000e+02],
                  [0, 0, 1]])

    # First and third frame of Kitti dataset
    img = cv2.imread('../data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread('../data/kitti/05/image_0/000002.png', cv2.IMREAD_GRAYSCALE)

elif DATASET == 'PARKING':
    # Parking
    K = np.array([[331.37, 0, 320],
                  [0, 369.568, 240],
                  [0, 0, 1]])

    # First and third frame of Kitti dataset
    img = cv2.imread('../data/parking/images/img_00000.png', cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread('../data/parking/images/img_00002.png', cv2.IMREAD_GRAYSCALE)

# img = cv2.imread('../data/data/0001.jpg', cv2.IMREAD_GRAYSCALE)
# img_2 = cv2.imread('../data/data/0002.jpg', cv2.IMREAD_GRAYSCALE)
# K = np.array([  [1379.74,   0,          760.35],
#                 [    0,     1382.08,    503.41],
#                 [    0,     0,          1 ]] )

# Select keypoints and descriptors of image 1 and 2
harris_scores = harris(img, corner_patch_size, harris_kappa)
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius) # [0,:]: row(y-loc in graph), [1,:] : column (x-loc in graph)
descriptors = describeKeypoints(img, keypoints, descriptor_radius)

dkp = np.zeros_like(keypoints)
keep = np.ones((keypoints.shape[1],)).astype('bool')
for j in range(keypoints.shape[1]):
    kptd, k = trackKLTRobustly(img, img_2, np.flipud(keypoints[:,j].T), r_T, n_iter, threshold)
    dkp[:, j] = kptd
    keep[j] = k


harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)

# Match descriptors between first two images
matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

# Plot keypoints matches
plt.clf()
plt.close()
plt.imshow(img_2, cmap='gray')
plt.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
plotMatches(matches, keypoints_2, keypoints)
plt.tight_layout()
plt.axis('off')
plt.show()

query_indices = np.nonzero(matches >= 0)[0] # keypoints 2 
database_indices = matches[query_indices] # keypoints 1 

matched_keypoints1 = keypoints[:, database_indices]
matched_keypoints2 = keypoints_2[:, query_indices]

# matched_keypoints1 = keypoints[:, keep]
# matched_keypoints2 = keypoints + dkp
# matched_keypoints2 = matched_keypoints2[:, keep]

# np.r_[matched_keypoints1[1,:], matched_keypoints1[0,inliers], np.ones((1, inliers.sum()))]



rng = np.random.default_rng(random_seed)

## Apply Ransac to Obtain the Essential Matrix and inliers // skimage library
model, inliers = ransac((matched_keypoints1.T, matched_keypoints2.T), FundamentalMatrixTransform, min_samples=8,
# model, inliers = ransac((matched_keypoints1.T, matched_keypoints2.T), EssentialMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000,
                        random_state=rng)

# F, inliers = ransacFundamentalMatrix(matched_keypoints1, matched_keypoints2)

p1 = np.r_[matched_keypoints1[None,1,inliers], matched_keypoints1[None,0,inliers], np.ones((1, inliers.sum()))]
p2 = np.r_[matched_keypoints2[None,1,inliers], matched_keypoints2[None,0,inliers], np.ones((1, inliers.sum()))]

# p1, _ = normalise2DPts(p1)
# p2, _ = normalise2DPts(p2)

# # Load outlier-free point correspondences
# p1 = np.loadtxt('../data/data/matches0001.txt')
# p2 = np.loadtxt('../data/data/matches0002.txt')

# p1 = np.r_[p1, np.ones((1, p1.shape[1]))]
# p2 = np.r_[p2, np.ones((1, p2.shape[1]))]

F = model.params

# F = T2.T @ F @ T1

# E = np.linalg.inv(K).T @ F @ np.linalg.inv(K)
E = K.T @ F @ K

# E = estimateEssentialMatrix(p1, p2, K, K)
# F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

Rots, u3 = decomposeEssentialMatrix(E)

R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)

M1 = K @ np.eye(3, 4)
M2 = K @ np.c_[R_C2_W, T_C2_W]
P = linearTriangulation(p1, p2, M1, M2)

# R_C_P, T_C_P, inlier_mask, _, _ = ransacLocalization(np.flipud(p1[:2,:]), P[:3,:].T, K)
# p1 = p1[:, inlier_mask]
# mask = (-5 < P[2,:]) & (P[2,:] < 5)
# P = P[:,mask]


# Visualize the 3-D scene
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


""" Questions
1. Why use F instead of E = E = K.T @ F @ K ? 
2. Camera 2 seems(?) right, but Camera 1 seems incorrect (pose).
3. match_lambda seems to have significant effect.


"""