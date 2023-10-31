import argparse
import os

import numpy as np

from fidgetyfind.constants import FEET_LIMBS, HANDS_LIMBS
from fidgetyfind.distal import get_distal_windows_entropy
from fidgetyfind.proximal import get_proximal_windows_entropy
from fidgetyfind.skeleton_smoothing import smooth_skeletons
from fidgetyfind.skin_and_flow import get_background_flow, get_flow_features, get_scoreable_windows_wrt_flow
from fidgetyfind.video_utils import get_video_fps


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Run FidgetyFind on a Single Video.')
    parser.add_argument('--video_filepath', type=str,
                        help='Path to video.')
    parser.add_argument('--skeletons_dir', type=str,
                        help='Path to directory containing skeleton detected by OpenPose.')
    parser.add_argument('--save_root_dir', type=str,
                        help='If specified, save the extracted features there.')
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    hips_entropies, hands_entropies, feet_entropies = run_fidgetyfind(
        video_filepath=args.video_filepath,
        skeletons_dir=args.skeletons_dir,
    )
    if args.save_root_dir is not None:
        video_id = os.path.basename(args.video_filepath).split(sep='.')[0]
        save_dir = os.path.join(args.save_root_dir, video_id)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'hips.npy'), arr=hips_entropies)
        np.save(os.path.join(save_dir, 'hands.npy'), arr=hands_entropies)
        np.save(os.path.join(save_dir, 'feet.npy'), arr=feet_entropies)


def run_fidgetyfind(video_filepath, skeletons_dir):
    smoothed_skeletons = smooth_skeletons(skeletons_dir)
    background_flow = get_background_flow(smoothed_skeletons, video_filepath)
    video_fps = get_video_fps(video_filepath)
    # Proximal
    # Hips
    scoreable_proximal_windows_wrt_bkg_flow = get_scoreable_windows_wrt_flow(
        background_flow=background_flow,
        background_flow_thresh=0.80,
        start_frame=100,
        window_length=50,
        window_stride=20,
    )
    scoreable_proximal_windows_wrt_bkg_flow = np.tile(scoreable_proximal_windows_wrt_bkg_flow.reshape(-1, 1), 2)
    hips_entropies = get_proximal_windows_entropy(
        smoothed_skeletons=smoothed_skeletons[..., :2],
        skeletons_detection_confidence=smoothed_skeletons[..., 2:],
        start_frame=100,
        window_length=50,
        window_stride=20,
        video_fps=video_fps,
        skel_lowconf_rate_threshold=0.1,
        large_motion_rate_threshold=0.2,
        in_range_rate_threshold=0.2,
        num_hist_bins=8,
        large_motion_threshold=10.0,
        minr=4.50,
        maxr=8.00,
    )
    hips_entropies = np.where(scoreable_proximal_windows_wrt_bkg_flow == 0, np.nan, hips_entropies)
    # Distal
    flow_features, distal_joints = get_flow_features(
        smoothed_skeletons=smoothed_skeletons,
        video_filepath=video_filepath,
    )
    scoreable_distal_windows_wrt_bkg_flow = get_scoreable_windows_wrt_flow(
        background_flow=background_flow,
        background_flow_thresh=1.75,
        start_frame=100,
        window_length=50,
        window_stride=20,
    )
    scoreable_distal_windows_wrt_bkg_flow = np.tile(scoreable_distal_windows_wrt_bkg_flow.reshape(-1, 1), 2)
    # Hands
    hands_entropies = get_distal_windows_entropy(
        smoothed_skeletons=smoothed_skeletons[..., :2],
        skeletons_detection_confidence=smoothed_skeletons[..., 2:],
        distal_limbs=HANDS_LIMBS,
        flow_features=[ff[2:] for ff in flow_features],
        distal_joints=distal_joints[:, 2:],
        flow_rel_mag_low_threshold=0.08,
        flow_rel_mag_high_threshold=None,
        start_frame=100,
        window_length=50,
        window_stride=20,
        skel_lowconf_threshold=0.35,
        skel_lowconf_rate_threshold=0.20,
        large_parent_motion_threshold=1.00,
        large_parent_motion_rate_threshold=0.30,
        num_hist_bins=16,
    )
    hands_entropies = np.where(scoreable_distal_windows_wrt_bkg_flow == 0, np.nan, hands_entropies)
    # Feet
    feet_entropies = get_distal_windows_entropy(
        smoothed_skeletons=smoothed_skeletons[..., :2],
        skeletons_detection_confidence=smoothed_skeletons[..., 2:],
        distal_limbs=FEET_LIMBS,
        flow_features=[ff[:2] for ff in flow_features],
        distal_joints=distal_joints[:, :2],
        flow_rel_mag_low_threshold=0.08,
        flow_rel_mag_high_threshold=None,
        start_frame=100,
        window_length=50,
        window_stride=20,
        skel_lowconf_threshold=0.35,
        skel_lowconf_rate_threshold=0.20,
        large_parent_motion_threshold=2.50,
        large_parent_motion_rate_threshold=0.10,
        num_hist_bins=16,
    )
    feet_entropies = np.where(scoreable_distal_windows_wrt_bkg_flow == 0, np.nan, feet_entropies)
    return hips_entropies, hands_entropies, feet_entropies


if __name__ == '__main__':
    main()
