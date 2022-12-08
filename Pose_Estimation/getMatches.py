import matplotlib.pyplot as plt
import numpy as np


def getMatches(matches, query_keypoints, database_keypoints):
    query_indices = np.nonzero(matches >= 0)[0]
    match_indices = matches[query_indices]

    x_from = np.array(query_keypoints[0, query_indices])
    x_to = np.array(database_keypoints[0, match_indices])
    y_from = np.array(query_keypoints[1, query_indices])
    y_to = np.array(database_keypoints[1, match_indices])

    p1 = np.array([y_from, x_from])
    p1 = np.r_[p1, np.ones((1, p1.shape[1]))]

    p2 = np.array([y_to, x_to])
    p2 = np.r_[p2, np.ones((1, p2.shape[1]))]

    return p1, p2





    # return