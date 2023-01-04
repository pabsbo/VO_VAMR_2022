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
num_candidate_keypoints = 50 #1000
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 5

# KLT parameters
r_T = 15
n_iter = 50
threshold = 0.1

# Re-triangulation parameters
baseline_threshold = 1.0

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
    R_C_W, T_C_W, inlier_mask, _, _ = ransacLocalization(np.flipud(matched_curr_keypoints), matched_landmarks.T, K)

    curr_state.keypoints = matched_curr_keypoints[:, inlier_mask]
    curr_state.landmarks = matched_landmarks[:, inlier_mask]

    ## Part 3. Triangulating new landmarks
    harris_scores = harris(curr_img, corner_patch_size, harris_kappa)
    candidate_keypoints = selectKeypoints(harris_scores, num_candidate_keypoints, nonmaximum_supression_radius) # [0,:]: row(y-loc in graph), [1,:] : column (x-loc in graph)
    candidate_keypoints = np.flipud(candidate_keypoints) # now (u,v)

    # Choose keypoints NOT redundant with curr_keypoints / existing candidate_keypoints
    in_curr_keypoints = (np.linalg.norm(np.repeat(matched_curr_keypoints, candidate_keypoints.shape[1], axis=1) - np.tile(candidate_keypoints, matched_curr_keypoints.shape[1]), axis=0) < 5).nonzero()[0] % candidate_keypoints.shape[1]
    in_curr_keypoints = np.unique(in_curr_keypoints)
    if prev_state.candidate_keypoints is not None:
        dkp = np.zeros_like(prev_state.candidate_keypoints)
        keep = np.ones((prev_state.candidate_keypoints.shape[1],)).astype('bool')
        for j in range(prev_state.candidate_keypoints.shape[1]):
            kptd, k = trackKLTRobustly(prev_img, curr_img, prev_state.candidate_keypoints[:,j].T, r_T, n_iter, threshold)
            dkp[:, j] = kptd
            keep[j] = k
        prev_candidate_in_curr_img = prev_state.candidate_keypoints + dkp
        prev_candidate_in_curr_img = prev_candidate_in_curr_img[:, keep]
        in_prev_candidate = (np.linalg.norm(np.repeat(prev_candidate_in_curr_img, candidate_keypoints.shape[1], axis=1) - np.tile(candidate_keypoints, prev_candidate_in_curr_img.shape[1]), axis=0) < 5).nonzero()[0] % candidate_keypoints.shape[1]
        in_prev_candidate = np.unique(in_prev_candidate)
        invalid_candidate_keypoints = np.unique(np.r_[in_curr_keypoints, in_prev_candidate])

        curr_state.candidate_keypoints = prev_candidate_in_curr_img
        curr_state.first_obs_keypoints = curr_state.first_obs_keypoints[:, keep]
        curr_state.first_obs_poses = curr_state.first_obs_poses[:, keep]
        curr_state.first_obs_position = curr_state.first_obs_position[:, keep]
    else:
        invalid_candidate_keypoints = in_curr_keypoints

    valid_candidate_keypoints = np.ones(candidate_keypoints.shape[1], dtype=np.bool8)
    valid_candidate_keypoints[invalid_candidate_keypoints] = False

    curr_state.extendCandidateKeypoints(candidate_keypoints[:,valid_candidate_keypoints])
    curr_state.extendFirstObsKeypoints(candidate_keypoints[:,valid_candidate_keypoints])
    curr_state.extendFirstObsPoses(np.r_[R_C_W.flatten(), T_C_W], valid_candidate_keypoints.sum())

    curr_camera_position = -np.matmul(R_C_W.T, T_C_W)
    curr_state.extendFirstObsPosition(curr_camera_position, valid_candidate_keypoints.sum())

    # Assigning more keypoints, landmarks: Triangulate if angle > threshold
    baselines = np.linalg.norm(curr_state.first_obs_position - curr_camera_position[:,None], axis=0)
    valid_baselines = np.unique(baselines)[np.unique(baselines) > baseline_threshold]

    total_triangulation_mask = np.zeros(len(baselines), dtype=np.bool8)
    for valid_baseline in valid_baselines:
        triangulation_mask = (valid_baseline == baselines)

        p1 = np.r_[curr_state.candidate_keypoints[:,triangulation_mask], np.ones((1, triangulation_mask.sum()))]
        p2 = np.r_[curr_state.first_obs_keypoints[:,triangulation_mask], np.ones((1, triangulation_mask.sum()))]
        first_obs_pose = curr_state.first_obs_poses[:,triangulation_mask][:,0]
        M1 = K @ np.c_[first_obs_pose[:9].reshape(3,3), first_obs_pose[9:].reshape(3,1)]
        M2 = K @ np.c_[R_C_W, T_C_W]
        P = linearTriangulation(p1, p2, M1, M2)

        curr_state.extendKeypoints(curr_state.candidate_keypoints[:,triangulation_mask])
        curr_state.extendLandmarks(P[:3,:])
        total_triangulation_mask |= triangulation_mask

    curr_state.candidate_keypoints = curr_state.candidate_keypoints[:,~total_triangulation_mask]
    curr_state.first_obs_keypoints = curr_state.first_obs_keypoints[:,~total_triangulation_mask]
    curr_state.first_obs_poses = curr_state.first_obs_poses[:,~total_triangulation_mask]
    curr_state.first_obs_position = curr_state.first_obs_position[:,~total_triangulation_mask]

    return curr_state, R_C_W, T_C_W, inlier_mask, 