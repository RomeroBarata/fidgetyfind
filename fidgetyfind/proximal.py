import numpy as np

from fidgetyfind.constants import PROXIMAL_LOWER_LIMBS_INDICES, STANDARD_FPS, LIMB_SCORE_THRESH, EPS
from fidgetyfind.geometry import get_angle_between, get_reference_length


def get_proximal_windows_entropy(smoothed_skeletons: np.ndarray, skeletons_detection_confidence: np.ndarray,
                                 start_frame: int, window_length: int, window_stride: int,
                                 video_fps: float,
                                 skel_lowconf_rate_threshold: float,
                                 large_motion_rate_threshold: float,
                                 in_range_rate_threshold: float,
                                 num_hist_bins: int,
                                 large_motion_threshold: float,
                                 minr: float,
                                 maxr: float) -> np.ndarray:
    """Compute FidgetyFind feature for proximal joints.

    Argument(s):
        smoothed_skeletons - numpy array of shape (num_frames, num_joints, 2).
        skeletons_detection_confidence - numpy array of shape (num_frames, num_joints, 1).
        start_frame - The initial frame to consider. Useful to skip the first few seconds of the video, where the
          recorder is still adjusting the camera.
        window_length - The length, in frames, of the considered window.
        window_stride - The stride length, in frames, between consecutive windows.
        video_fps - The FPS of the video.
        skel_lowconf_rate_threshold - Threshold for the ratio of low-confidence joint detections that are acceptable.
        large_motion_rate_threshold - Threshold for the ratio of large movements that are acceptable.
        in_range_rate_threshold - Threshold for the ratio of movements that must be within the specified
          magnitude range.
        num_hist_bins - Number of bins used to build the histogram over the displacement relative angles.
        large_motion_threshold - Threshold used to decide whether a movement is large or not.
        minr - Only movements with a magnitude above minr are considered.
        maxr - Only movements with a magnitude below maxr are considered.
    Return(s):
        A numpy array of shape (num_windows, 2) containing the movement entropy for the right and left lower proximal
        joints (the hips). NaN values mean the method failed for associated windows.
    """
    motion_features = get_proximal_motion_features(smoothed_skeletons, skeletons_detection_confidence)
    windows_motion_entropy = get_proximal_windows_motion_entropy(
        motion_features=motion_features,
        start_frame=start_frame,
        window_length=window_length,
        window_stride=window_stride,
        skel_lowconf_rate_threshold=skel_lowconf_rate_threshold,
        video_fps=video_fps,
        large_motion_threshold=large_motion_threshold,
        large_motion_rate_threshold=large_motion_rate_threshold,
        minr=minr, maxr=maxr,
        in_range_rate_threshold=in_range_rate_threshold,
        num_hist_bins=num_hist_bins,
    )
    proximal_windows_entropy = windows_motion_entropy[..., 0] / np.log(num_hist_bins)
    return proximal_windows_entropy


def get_proximal_motion_features(smoothed_skeletons: np.ndarray,
                                 skeletons_detection_confidence: np.ndarray) -> np.ndarray:
    num_frames = smoothed_skeletons.shape[0]
    motion_features = np.zeros([num_frames - 1, len(PROXIMAL_LOWER_LIMBS_INDICES), 3])
    for t in range(num_frames - 1):
        skeleton_t = smoothed_skeletons[t]
        score_t = skeletons_detection_confidence[t]
        skeleton_tp1 = smoothed_skeletons[t + 1]
        score_tp1 = skeletons_detection_confidence[t + 1]
        for limb_index, limb in enumerate(PROXIMAL_LOWER_LIMBS_INDICES):
            score = np.min([
                score_t[limb[1]], score_t[limb[2]],
                score_tp1[limb[1]], score_tp1[limb[2]],
            ])
            if score <= 0:
                motion_features[t, limb_index, :] = np.array([0.0, 0.0, score])
            else:
                v = skeleton_tp1[limb[2]] - skeleton_t[limb[2]]
                ref_len = get_reference_length(skeleton_t, reference_joints_indices=[limb[2], limb[1]])
                mag = 100 * np.linalg.norm(v) / ref_len
                angle = get_angle_between(skeleton_tp1[limb[2]] - skeleton_tp1[limb[1]], v)
                motion_features[t, limb_index, :] = np.array([angle, mag, score])
    return motion_features


def get_proximal_windows_motion_entropy(motion_features: np.ndarray, start_frame: int, window_length: int,
                                        window_stride: int,
                                        skel_lowconf_rate_threshold: float, video_fps: float,
                                        large_motion_threshold: float,
                                        large_motion_rate_threshold: float, minr: float, maxr: float,
                                        in_range_rate_threshold: float, num_hist_bins: int) -> np.ndarray:
    num_frames = motion_features.shape[0] + 1
    win_starts = range(start_frame, num_frames - window_length, window_stride)
    num_limbs = motion_features.shape[1]
    windows_entropy = np.full([len(win_starts), num_limbs, 1], fill_value=np.nan, dtype=np.float32)
    for window_index, window_start_frame in enumerate(win_starts):  # for each window
        mo_win = motion_features[window_start_frame:window_start_frame + window_length]
        window_entropies = np.full([num_limbs, 1], fill_value=np.nan, dtype=np.float32)
        for limb_index in range(num_limbs):
            # Process motion features
            limb_mo_win = mo_win[:, limb_index, :]
            low_conf_ratio = np.sum(limb_mo_win[:, 2] < LIMB_SCORE_THRESH) / (limb_mo_win.shape[0])
            if low_conf_ratio > skel_lowconf_rate_threshold:
                continue
            a_raw, r_raw, s_raw = limb_mo_win.T
            if abs(video_fps - STANDARD_FPS) > 1.0:
                r_raw = r_raw * video_fps / STANDARD_FPS
            # If there are mostly large movement, mark it as a bad window
            large_motion_rate = np.sum((r_raw > large_motion_threshold)) / r_raw.size
            if large_motion_rate > large_motion_rate_threshold:
                continue
            a, r, s = post_process_ang_mag_score(a_raw, r_raw, s_raw, minr, maxr)
            # If the motions are mild, but there are too few in the range, give it zeros score
            in_range_rate = np.sum(r > 0.0) / r_raw.size
            if in_range_rate < in_range_rate_threshold:
                win_fea_l = 0.0
            else:
                a1 = a[a != 0.0]
                if a1.size != 0:
                    a = a1
                ha = np.histogram(a, bins=num_hist_bins, range=(-np.pi, np.pi), density=True)
                ha0 = ha[0] / ha[0].sum() + EPS
                ea = -(ha0 * np.log(np.abs(ha0))).sum()
                win_fea_l = ea
                if np.isnan(win_fea_l).any():
                    win_fea_l = 0.0
            window_entropies[limb_index] = win_fea_l
        windows_entropy[window_index] = window_entropies
    return windows_entropy


def post_process_ang_mag_score(a, r, s, min_mag: float, max_mag: float):
    a1 = a.copy()
    r1 = r.copy()
    s1 = s.copy()

    s1[s < -1] = -1
    r1[(s < 0) + (r < min_mag) + (r > max_mag)] = 0
    a1[(s < 0) + (r < min_mag) + (r > max_mag)] = 0

    return a1, r1, s1
