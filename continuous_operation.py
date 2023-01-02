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
from Ex7_ransac.ransacLocalization import ransacLocalization
from Ex8_KLT.track_klt_robustly import trackKLTRobustly

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from state import State

# KLT parameters
r_T = 15
n_iter = 50
threshold = 0.1

def processFrame(curr_img, prev_img, prev_state : State, K):
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

    curr_state = State(matched_curr_keypoints[:, inlier_mask], matched_landmarks[:, inlier_mask])


    return curr_state, R_C_P, T_C_P, inlier_mask