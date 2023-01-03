import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

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
from Ex7_ransac.ransacLocalization import ransacLocalization
from Ex8_KLT.track_klt_robustly import trackKLTRobustly

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from state import State

# Parameters used in previous exercises
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200 #1000
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 5

# KLT parameters
r_T = 15
n_iter = 50
threshold = 0.1

def processFrame(curr_img, prev_img, prev_state : State, K):
    curr_state = copy.deepcopy(prev_state)

    ## Part 1. Associating keypoints to existing landmarks
    dkp = np.zeros_like(prev_state.keypoints)
    keep = np.ones((prev_state.keypoints.shape[1],)).astype('bool')
    for j in range(prev_state.keypoints.shape[1]):
        kptd, k = trackKLTRobustly(prev_img, curr_img, prev_state.keypoints[:,j].T, r_T, n_iter, threshold)
        dkp[:, j] = kptd
        keep[j] = k

    matched_prev_keypoints = prev_state.keypoints[:, keep]
    matched_curr_keypoints = prev_state.keypoints + dkp
    matched_curr_keypoints = matched_curr_keypoints[:, keep]
    matched_landmarks = prev_state.landmarks[:, keep]

    ## Part 2. Estimating the current pose
    R_C_P, T_C_P, inlier_mask, _, _ = ransacLocalization(np.flipud(matched_curr_keypoints), matched_landmarks.T, K)

    ## Part 3. Triangulating new landmarks
    harris_scores = harris(curr_img, corner_patch_size, harris_kappa)
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius) # [0,:]: row(y-loc in graph), [1,:] : column (x-loc in graph)
    keypoints = np.flipud(keypoints)

    # Choose keypoints NOT redundant with curr_keypoints / existing candidate_keypoints
    not_curr_keypoints = (np.linalg.norm(np.repeat(matched_curr_keypoints, keypoints.shape[1], axis=1) - np.tile(keypoints, matched_curr_keypoints.shape[1]), axis=0) > 5).nonzero()[0] % keypoints.shape[1]
    if prev_state.candidate_keypoints is not None: # TODO: prev candidate progate! by klt tracking!!
        dkp = np.zeros_like(prev_state.candidate_keypoints)
        keep = np.ones((prev_state.candidate_keypoints.shape[1],)).astype('bool')
        for j in range(prev_state.candidate_keypoints.shape[1]):
            kptd, k = trackKLTRobustly(prev_img, curr_img, prev_state.candidate_keypoints[:,j].T, r_T, n_iter, threshold)
            dkp[:, j] = kptd
            keep[j] = k
        prev_candidate_in_curr_img = prev_state.candidate_keypoints + dkp
        prev_candidate_in_curr_img = prev_candidate_in_curr_img[:, keep]
        not_prev_candidate = (np.linalg.norm(np.repeat(prev_candidate_in_curr_img, keypoints.shape[1], axis=1) - np.tile(keypoints, prev_candidate_in_curr_img.shape[1]), axis=0) > 5).nonzero()[0] % keypoints.shape[1]
        candidate_keypoints_idx = not_curr_keypoints & not_prev_candidate

        curr_state.candidate_keypoints = curr_state.candidate_keypoints[:, keep]
        curr_state.first_obs_keypoints = curr_state.first_obs_keypoints[:, keep]
        curr_state.first_obs_poses = curr_state.first_obs_poses[:, keep]
    else:
        candidate_keypoints_idx = not_curr_keypoints

    # candidate_keypoints_idx = not_curr_keypoints & not_prev_candidate

    curr_state.keypoints = matched_curr_keypoints[:, inlier_mask]
    curr_state.landmarks = matched_landmarks[:, inlier_mask]
    # curr_state = State(matched_curr_keypoints[:, inlier_mask], 
    #                    matched_landmarks[:, inlier_mask], 
    #                    keypoints[:,candidate_keypoints_idx])

    return curr_state, R_C_P, T_C_P, inlier_mask, 