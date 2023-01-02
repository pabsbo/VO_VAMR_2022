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
from Ex7_ransac.ransacFundamentalMatrix import ransacFundamentalMatrix


from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

def initialization(dataset): # dataset = 'KITTI', 'PARKING'
    # Parameters used in previous exercises
    corner_patch_size = 9
    harris_kappa = 0.08
    num_keypoints = 200 #1000
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 5

    random_seed = 9

    if dataset == 'KITTI':
        # KITTI
        img_path = '../data/kitti/05/image_0'
        K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])

        # First and third frame of Kitti dataset
        img = cv2.imread(f'{img_path}/000000.png', cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(f'{img_path}/000002.png', cv2.IMREAD_GRAYSCALE)

    elif dataset == 'PARKING':
        # Parking
        img_path = '../data/parking/images'
        K = np.array([[331.37, 0, 320],
                    [0, 369.568, 240],
                    [0, 0, 1]])

        # First and third frame of Kitti dataset
        img = cv2.imread(f'.{img_path}/img_00000.png', cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(f'{img_path}/img_00002.png', cv2.IMREAD_GRAYSCALE)

    # Select keypoints and descriptors of image 1 and 2
    harris_scores = harris(img, corner_patch_size, harris_kappa)
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius) # [0,:]: row(y-loc in graph), [1,:] : column (x-loc in graph)
    descriptors = describeKeypoints(img, keypoints, descriptor_radius)

    harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
    keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
    descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)

    # Match descriptors between first two images
    matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

    query_indices = np.nonzero(matches >= 0)[0] # keypoints 2 
    database_indices = matches[query_indices] # keypoints 1 

    matched_keypoints1 = keypoints[:, database_indices]
    matched_keypoints2 = keypoints_2[:, query_indices]

    rng = np.random.default_rng(random_seed)

    ## Apply Ransac to Obtain the Essential Matrix and inliers // skimage library
    model, inliers = ransac((matched_keypoints1.T, matched_keypoints2.T), FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=5000,
                            random_state=rng)

    p1 = np.r_[matched_keypoints1[None,1,inliers], matched_keypoints1[None,0,inliers], np.ones((1, inliers.sum()))]
    p2 = np.r_[matched_keypoints2[None,1,inliers], matched_keypoints2[None,0,inliers], np.ones((1, inliers.sum()))]

    F = model.params

    Rots, u3 = decomposeEssentialMatrix(F)

    R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)

    M1 = K @ np.eye(3, 4)
    M2 = K @ np.c_[R_C2_W, T_C2_W]
    P = linearTriangulation(p1, p2, M1, M2)

    
    return p1[:2,:], P[:3,:], R_C2_W, T_C2_W, img  # p1: keypoints in frame 0, P: 3D landmarks, R_C2_W: camera rotation matrix, T_C2_W: camera position, img: the first frame

""" Questions
1. Why do we return R_C2_W? Initial camera pose is just (0, 0, 1), isn't it ?
On the document, BOOTSTRAP the initial camera poses and landmarks,,,?


"""