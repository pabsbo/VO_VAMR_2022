""" Data Structure """
import numpy as np
class State:
    def __init__(self, keypoints=None, landmarks=None, 
                       candidate_keypoints=None, first_obs_keypoints=None, first_obs_poses=None):
                       
        self.keypoints = keypoints # P: 2D keypoints [2, K] *first row: u (x), second row: v (y)*
        self.landmarks = landmarks # X: corresponding 3D landmakrs [3, K]
        self.candidate_keypoints = candidate_keypoints # C: set of candidate keypoints [2, M]
        self.first_obs_keypoints = first_obs_keypoints # F: first observations of the track of keypoints [2, M]
        self.first_obs_poses = first_obs_poses # T: camera poses at the first observations of keypoints [12, M]

    def extendKeypoints(self, candidate_keypoints):
        # candidate_keypoints: [2, N']
        return np.append(self.keypoints, candidate_keypoints, axis=1)

    def extendCandidateKeypoints(self, keypoints):
        # keypoints: [2, M']
        return np.append(self.candidate_keypoints, keypoints, axis=1)