# import cv2
# import numpy as np
#
# from Pose_Estimation.decompose_essential_matrix import decomposeEssentialMatrix
# from Pose_Estimation.disambiguate_relative_pose import disambiguateRelativePose
# from Pose_Estimation.estimate_pose_dlt import estimatePoseDLT
# from Pose_Estimation.linear_triangulation import linearTriangulation
# from Pose_Estimation.projectPoints import projectPoints
# from Pose_Estimation.estimate_essential_matrix import estimateEssentialMatrix
#
#
# def ransacLocalization(p1, p2, K):
#     """
#     best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
#     False if the match is an outlier, True otherwise.
#     """
#
#     num_iterations = 1000
#     pixel_tolerance = 10
#     k = 8
#
#     # Initialize RANSAC
#
#     best_inlier_mask = np.zeros(len(p1))
#     # (row, col) to (u, v)
#     # matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)
#     max_num_inliers_history = []
#     num_iteration_history = []
#     max_num_inliers = 0
#
#     # RANSAC
#     i = 0
#     while num_iterations > i:
#         # Model from k samples (DLT or P3P)
#         shuffler = np.random.permutation(p1.shape[0])[:k]
#         p1_sample = p1[shuffler[:k]]
#         p2_sample = p2[shuffler[:k]]
#         p1_test = p1[shuffler[k:]]
#         p2_test = p2[shuffler[k:]]
#
#
#         E = estimateEssentialMatrix(p1_sample, p2_sample, K, K)
#
#         Rots, u3 = decomposeEssentialMatrix(E)
#
#         R_C2_W, T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)
#         M1 = K @ np.eye(3, 4)
#         M2 = K @ np.c_[R_C2_W, T_C2_W]
#         P = linearTriangulation(p1, p2, M1, M2)
#
#
#         # Count inliers
#         if not use_p3p:
#             C_landmarks = np.matmul(R_C_W_guess, corresponding_landmarks[:, :, None]).squeeze(-1) + t_C_W_guess[None, :]
#             projected_points = projectPoints(C_landmarks, K)
#             difference = matched_query_keypoints - projected_points.T
#             errors = (difference ** 2).sum(0)
#             is_inlier = errors < pixel_tolerance ** 2
#
#         else:
#             # If we use p3p, also consider inliers for the 4 solutions.
#             is_inlier = np.zeros(corresponding_landmarks.shape[0])
#             for alt_idx in range(len(R_C_W_guess)):
#                 C_landmarks = np.matmul(R_C_W_guess[alt_idx], corresponding_landmarks[:, :, None]).squeeze(-1) + \
#                               t_C_W_guess[alt_idx][None, :].squeeze(-1)
#                 projected_points = projectPoints(C_landmarks, K)
#                 difference = matched_query_keypoints - projected_points.T
#                 errors = (difference ** 2).sum(0)
#                 alternative_is_inlier = errors < pixel_tolerance ** 2
#                 if alternative_is_inlier.sum() > is_inlier.sum():
#                     is_inlier = alternative_is_inlier
#
#         min_inlier_count = 30 if tweaked_for_more else 6
#
#         if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
#             max_num_inliers = is_inlier.sum()
#             best_inlier_mask = is_inlier
#
#         if adaptive:
#             # estimate of the outlier ratio
#             outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]
#             # formula to compute number of iterations from estimated outlier ratio
#             confidence = 0.95
#             upper_bound_on_outlier_ratio = 0.90
#             outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
#             num_iterations = np.log(1 - confidence) / np.log(1 - (1 - outlier_ratio) ** k)
#             # cap the number of iterations at 15000
#             num_iterations = min(15000, num_iterations)
#
#         num_iteration_history.append(num_iterations)
#         max_num_inliers_history.append(max_num_inliers)
#
#         i += 1
#
#     if max_num_inliers == 0:
#         R_C_W = None
#         t_C_W = None
#     else:
#         M_C_W = estimatePoseDLT(matched_query_keypoints[:, best_inlier_mask].T,
#                                 corresponding_landmarks[best_inlier_mask, :], K)
#         R_C_W = M_C_W[:, :3]
#         t_C_W = M_C_W[:, -1]
#
#         if adaptive:
#             print("    Adaptive RANSAC: Needed {} iteration to converge.".format(i - 1))
#             print("    Adaptive RANSAC: Estimated Ouliers: {} %".format(100 * outlier_ratio))
#
#     return R_C_W, t_C_W, best_inlier_mask, max_num_inliers_history, num_iteration_history
