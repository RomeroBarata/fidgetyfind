import math
from typing import Optional

import numpy as np

from fidgetyfind.constants import NECK_AND_MID_HIP_INDICES
from fidgetyfind.geometry import get_reference_length


def get_distal_windows_entropy(smoothed_skeletons: np.ndarray, skeletons_detection_confidence: np.ndarray,
                               distal_limbs: list[list[int]],
                               flow_features: list[list[tuple[np.ndarray, np.ndarray]]],
                               distal_joints: np.ndarray,
                               flow_rel_mag_low_threshold: Optional[float],
                               flow_rel_mag_high_threshold: Optional[float],
                               start_frame: int,
                               window_length: int,
                               window_stride: int,
                               skel_lowconf_threshold: float,
                               skel_lowconf_rate_threshold: float,
                               large_parent_motion_threshold: float,
                               large_parent_motion_rate_threshold: float,
                               num_hist_bins: int,
                               ) -> np.ndarray:
    """Compute FidgetFind feature for distal joints.

    Argument(s):
        smoothed_skeletons - numpy array of shape (num_frames, num_joints, 2).
        skeletons_detection_confidence - numpy array of shape (num_frames, num_joints, 1).
        distal_limbs - A two-element list containing the indices for the right and left sides of the distal joint.
        flow_features - A list, of length num_frames - 1, containing the flow features for the selected pixels
          (i.e. segmented joint).
        distal_joints - numpy array of shape (num_frames - 1, num_distal_limbs, 3) containing the centroid and the
          confidence for the segmented distal limbs. The confidence is just the confidence of the detected extremity.
        flow_rel_mag_low_threshold - Threshold for considering flow with minimum magnitude above the threshold.
        flow_rel_mag_high_threshold - Threshold for considering flow with minimum magnitude below the threshold.
        start_frame - The initial frame to consider. Useful to skip the first few seconds of the video, where the
          recorder is still adjusting the camera.
        window_length - The length, in frames, of the considered window.
        window_stride - The stride length, in frames, between consecutive windows.
        skel_lowconf_threshold - Threshold to decide whether a detection is low quality or not.
        skel_lowconf_rate_threshold - Threshold for the ratio of low-confidence joint detections that is acceptable.
        large_parent_motion_threshold - Threshold to decide whether a motion is too large.
        large_parent_motion_rate_threshold - Threshold for the ratio of large movements that is acceptable.
        num_hist_bins - Number of bins used to build the histogram over the flow angles.
    Returns:
        A numpy array of shape (num_windows, 2) with the right/left entropies of the specified distal joint.
        NaN values mean the method failed for associated windows.
    """
    num_frames = len(smoothed_skeletons)
    num_limbs = distal_joints.shape[1]
    scores = np.full((num_frames - 1, num_limbs), fill_value=np.nan, dtype=np.float32)
    parent_motion = np.full((num_frames - 1, num_limbs), fill_value=np.nan, dtype=np.float32)
    orientation_histogram_per_frame = []
    for t in range(num_frames - 1):
        orientation_histogram_per_limb = {}
        for limb_index, limb in enumerate(distal_limbs):
            distal_joint = distal_joints[t, limb_index]
            distal_joint_score = distal_joint[-1:]
            score = np.min([skeletons_detection_confidence[t + 1, limb[1]],
                            skeletons_detection_confidence[t + 1, limb[0]],
                            distal_joint_score
                            ],
                           ).item()
            scores[t, limb_index] = score
            ptl_motion = np.linalg.norm(smoothed_skeletons[t + 1, limb[1]] - smoothed_skeletons[t, limb[1]])
            ptl_motion = 100 * ptl_motion / get_reference_length(smoothed_skeletons[t + 1],
                                                                 reference_joints_indices=NECK_AND_MID_HIP_INDICES,
                                                                 )
            parent_motion[t, limb_index] = ptl_motion
            if score <= 0:
                continue
            coords, flows = flow_features[t][limb_index]
            parent_limb = smoothed_skeletons[t, limb[1]] - smoothed_skeletons[t, limb[0]]
            flows = normalise_flow_wrt_parent_limb(flows, parent_limb)
            limb_orientation_histogram = orientation_histogram_from_flow(
                flows,
                bins=num_hist_bins,
                motion_low_threshold=flow_rel_mag_low_threshold,
                motion_high_threshold=flow_rel_mag_high_threshold,
            )[0]
            orientation_histogram_per_limb[limb_index] = limb_orientation_histogram
        orientation_histogram_per_frame.append(orientation_histogram_per_limb)
    # Build per window features/scores
    window_start_end_frames = [
        (window_start_frame, window_start_frame + window_length)
        for window_start_frame in range(start_frame,
                                        len(orientation_histogram_per_frame) - window_length + 1,
                                        window_stride,
                                        )
    ]
    num_windows = len(window_start_end_frames)
    distal_windows_entropy = np.full((num_windows, num_limbs), fill_value=np.nan, dtype=np.float32)
    for window_index, (window_start_frame, window_end_frame) in enumerate(window_start_end_frames):
        orientation_histogram_window = orientation_histogram_per_frame[window_start_frame:window_end_frame]
        scores_window = scores[window_start_frame:window_end_frame]
        parent_motion_window = parent_motion[window_start_frame:window_end_frame]
        for limb_index in range(len(distal_limbs)):
            # Execute checks to decide whether it is a bad window
            # # Average detection score in the window for the current limb (if too low we cannot score)
            low_conf_ratio = np.mean(scores_window[:, limb_index] < skel_lowconf_threshold)
            if low_conf_ratio > skel_lowconf_rate_threshold:
                continue
            # # Average motion of parent body part (if too high we cannot score)
            large_parent_motion = np.mean(parent_motion_window[:, limb_index] > large_parent_motion_threshold)
            if large_parent_motion > large_parent_motion_rate_threshold:
                continue
            limb_agg_orientation_hist = sum(
                limb_index2orientation_hist[limb_index]
                for limb_index2orientation_hist in orientation_histogram_window
                if limb_index in limb_index2orientation_hist
            )
            limb_agg_orientation_hist = limb_agg_orientation_hist / np.maximum(np.sum(limb_agg_orientation_hist), 1.0)
            limb_agg_orientation_hist_entropy = entropy(limb_agg_orientation_hist)
            limb_agg_orientation_hist_entropy /= math.log(num_hist_bins)
            distal_windows_entropy[window_index, limb_index] = limb_agg_orientation_hist_entropy
    return distal_windows_entropy


def normalise_flow_wrt_parent_limb(flows, parent_limb):
    parent_limb_norm = np.linalg.norm(parent_limb, axis=-1)
    flows = flows / parent_limb_norm
    return flows


def orientation_histogram_from_flow(flow, bins,
                                    motion_low_threshold: Optional[float],
                                    motion_high_threshold: Optional[float],
                                    ):
    gy, gx = flow.T  # flow is original gx, gy, but the one saved with mask is gy, gx
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)  # [-pi, pi]
    mask = np.full_like(orientation, fill_value=True, dtype=bool)
    if motion_low_threshold is not None:
        mask = mask & (magnitude > motion_low_threshold)
    if motion_high_threshold is not None:
        mask = mask & (magnitude < motion_high_threshold)
    orientation_histogram = np.histogram(orientation[mask], bins=bins, range=(-np.pi, np.pi))
    return orientation_histogram


def entropy(x, eps=1e-7):
    ent = np.sum(-(x * np.log(x + eps)))
    return ent
