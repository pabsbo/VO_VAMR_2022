""" Data Structure """
import numpy as np

class State:
    def __init__(self, keypoints=None, landmarks=None, 
                       candidate_keypoints=None, first_obs_keypoints=None, first_obs_poses=None,
                       first_obs_position=None):
                       
        self.keypoints = keypoints # P: 2D keypoints [2, K] *first row: u (x), second row: v (y)*
        self.landmarks = landmarks # X: corresponding 3D landmakrs [3, K]
        self.candidate_keypoints = candidate_keypoints # C: set of candidate keypoints [2, M]
        self.first_obs_keypoints = first_obs_keypoints # F: first observations of the track of keypoints [2, M]
        self.first_obs_poses = first_obs_poses # T: camera poses at the first observations of keypoints [12, M]
        self.first_obs_position = first_obs_position

    def extendKeypoints(self, keypoints):
        # candidate_keypoints: [2, K']
        self.keypoints = np.append(self.keypoints, keypoints, axis=1)

    def extendLandmarks(self, landmarks):
        # landmarks: [3, K']
        self.landmarks = np.append(self.landmarks, landmarks, axis=1)

    def extendCandidateKeypoints(self, candidate_keypoints):
        # candidate_keypoints: [2, M']
        if self.candidate_keypoints is None:
            self.candidate_keypoints = candidate_keypoints
        else:
            self.candidate_keypoints = np.append(self.candidate_keypoints, candidate_keypoints, axis=1)

    def extendFirstObsKeypoints(self, first_obs_keypoints):
        # first_obs_keypoints: [2, M']
        if self.first_obs_keypoints is None:
            self.first_obs_keypoints = first_obs_keypoints
        else:
            self.first_obs_keypoints = np.append(self.first_obs_keypoints, first_obs_keypoints, axis=1)

    def extendFirstObsPoses(self, first_obs_poses, size):
        # first_obs_poses: [12, ], size: M'
        first_obs_poses = np.tile(first_obs_poses, (size,1)).T
        if self.first_obs_poses is None:
            self.first_obs_poses = first_obs_poses
        else:
            self.first_obs_poses = np.append(self.first_obs_poses, first_obs_poses, axis=1)

    def extendFirstObsPosition(self, first_obs_position, size):
        # first_obs_poses: [3, ], size: M'
        first_obs_position = np.tile(first_obs_position, (size,1)).T
        if self.first_obs_position is None:
            self.first_obs_position = first_obs_position
        else:
            self.first_obs_position = np.append(self.first_obs_position, first_obs_position, axis=1)