NECK_AND_MID_HIP_INDICES = [1, 8]
PROXIMAL_LOWER_LIMBS_INDICES = [
    [8, 9, 10],  # mid hip -> right thigh
    [8, 12, 13],  # mid-hip -> left thigh
]
STANDARD_FPS = 30.0
JOINT_SCORE_THRESH = 0.1
LIMB_SCORE_THRESH = 0.1
EPS = 1e-5
FEET_LIMBS = [
    [10, 11, -100],  # right knee, right ankle, right foot (not detected by openpose)
    [13, 14, -100],  # left knee, left ankle, left foot (not detected by openpose)
]
HANDS_LIMBS = [
    [3, 4, -1],  # right elbow, right wrist, right hand (not detected by openpose)
    [6, 7, -1],  # left elbow, left wrist, left hand (not detected by openpose)
]
