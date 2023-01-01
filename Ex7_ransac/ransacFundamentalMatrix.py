import cv2
import numpy as np

from Ex1_augReal.projectPoints import projectPoints
from Ex2_dlt.estimate_pose_dlt import estimatePoseDLT
from Ex6_triangulation.fundamental_eight_point_normalized import fundamentalEightPointNormalized


def ransacFundamentalMatrix(matched_database_keypoints, matched_query_keypoints):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.

    matched_database_keypoints : n x 2,  [:,0] : row, [:,1] : column
    """
    num_iterations = float('inf')
    epsilon = 0.001
    k = 10

    # Initialize RANSAC
    F_estimate = np.zeros((3,3))
    F_best_estimate = np.zeros((3,3))
    best_inlier_mask = np.zeros(matched_query_keypoints.shape[0])
    # (row, col) to (u, v)
    matched_database_keypoints = np.flip(matched_database_keypoints, axis=0)
    matched_database_keypoints = np.r_[matched_database_keypoints, np.ones((1, matched_database_keypoints.shape[1]))]
    matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)
    matched_query_keypoints = np.r_[matched_query_keypoints, np.ones((1, matched_query_keypoints.shape[1]))]
    max_num_inliers = 0

    # RANSAC
    i = 0
    while num_iterations > i:
        # Model from k samples 
        indices = np.random.permutation(matched_database_keypoints.shape[1])[:k]
        database_sample = matched_database_keypoints[:, indices]
        query_sample = matched_query_keypoints[:, indices]

        F_estimate = fundamentalEightPointNormalized(database_sample, query_sample)

        # Count inliers
        errors = np.abs(np.diag(matched_query_keypoints.T @ F_estimate @ matched_database_keypoints))
        is_inlier = errors < epsilon

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= k:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier
            database_sample_inliers = matched_database_keypoints[:, best_inlier_mask]
            query_sample_inliers = matched_query_keypoints[:, best_inlier_mask]
            F_best_estimate = fundamentalEightPointNormalized(database_sample_inliers, query_sample_inliers)

        # estimate of the outlier ratio
        outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]
        # formula to compute number of iterations from estimated outlier ratio
        confidence = 0.95
        upper_bound_on_outlier_ratio = 0.90
        outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
        num_iterations = np.log(1-confidence)/np.log(1-(1-outlier_ratio)**k)
        # cap the number of iterations at 15000
        num_iterations = min(15000, num_iterations)

        i += 1

    return F_best_estimate, best_inlier_mask
