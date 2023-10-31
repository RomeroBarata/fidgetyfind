import cv2 as cv
import numpy as np

from fidgetyfind.constants import LIMB_SCORE_THRESH, JOINT_SCORE_THRESH, NECK_AND_MID_HIP_INDICES
from fidgetyfind.constants import FEET_LIMBS, HANDS_LIMBS
from fidgetyfind.geometry import rotate, get_reference_length


def get_flow_features(smoothed_skeletons: np.ndarray, video_filepath: str):
    """Function to identify flow of hands and feet."""
    ref_points = NECK_AND_MID_HIP_INDICES
    limbs = FEET_LIMBS + HANDS_LIMBS
    w_foot_per_trunk = 0.6
    l_foot_per_trunk = 0.6
    w_hand_per_trunk = 0.35
    l_hand_per_trunk = 0.35

    cap = cv.VideoCapture(video_filepath)
    num_frames = smoothed_skeletons.shape[0]
    distal_centroids = -np.ones([num_frames - 1, len(limbs), 3])
    flow_features = []
    _, previous_frame = cap.read()
    for t in range(num_frames - 1):
        prev_frame_grey = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
        _, frame = cap.read()
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_frame_grey, frame_grey, None,
                                           pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                           poly_n=5, poly_sigma=1.2, flags=0)
        previous_frame = frame
        skel = smoothed_skeletons[t]
        ref_len = get_reference_length(skel, ref_points)
        sample_size = int(ref_len * 0.03)
        trunk_length = ref_len
        frame_cs = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        flow_feature_per_limb = [None for _ in limbs]
        masks = [None for _ in limbs]  # each mask is for whole frame and identifies where the hand or foot is
        for limb_c, limb in enumerate(limbs):
            skel_mask = np.zeros(flow.shape[:2], dtype='uint8')
            score = np.min([skel[limb[0], 2], skel[limb[1], 2]]).item()
            if score < LIMB_SCORE_THRESH:
                flow_feature_per_limb[limb_c] = np.array([[], []])
                continue
            distal_parent = skel[limb[0], [1, 0]]  # cartesian to opencv
            distal_joint = skel[limb[1], [1, 0]]  # cartesian to opencv
            diff = distal_joint - distal_parent
            diff = diff / np.linalg.norm(diff)
            if limb_c < 2:  # feet
                # Note that the angles here are the same as conventional cartesian angles because of opencv coords.
                if limb_c == 0:  # right foot
                    foot_l_dir = rotate(diff, 120)
                    foot_w_dir = rotate(foot_l_dir, -90)
                else:  # limb_c == 1 ; left foot
                    foot_l_dir = rotate(diff, -120)
                    foot_w_dir = rotate(foot_l_dir, 90)

                c1 = distal_joint + foot_l_dir * trunk_length * w_foot_per_trunk * 0.75
                c2 = c1 + foot_w_dir * trunk_length * l_foot_per_trunk
                c4 = distal_joint - foot_l_dir * trunk_length * w_foot_per_trunk * 0.25
                c3 = c4 + foot_w_dir * trunk_length * l_foot_per_trunk
            else:  # hands
                # Note that the angles here are the same as conventional cartesian angles because of opencv coords.
                if limb_c == 2:  # right hand
                    hand_l_dir = rotate(diff, 120)
                    hand_w_dir = rotate(hand_l_dir, -90)
                else:  # limb_c == 3 ; left hand
                    hand_l_dir = rotate(diff, -120)
                    hand_w_dir = rotate(hand_l_dir, 90)
                c1 = distal_joint + hand_l_dir * trunk_length * w_hand_per_trunk * 0.5
                c2 = c1 + hand_w_dir * trunk_length * l_hand_per_trunk
                c4 = distal_joint - hand_l_dir * trunk_length * w_hand_per_trunk * 0.5
                c3 = c4 + hand_w_dir * trunk_length * l_hand_per_trunk

            cnt = np.int0(np.array([c1, c2, c3, c4]))
            cv.drawContours(skel_mask, [cnt], -1, 255, -1)

            # skin detector
            # 1 before 0 because we go back to numpy
            # We select a square region around the wrist or ankle. This is likely to fail in cases where the infant is
            # clothed. Maybe we could improve by guessing the center of the hand or foot. E.g. for the hand,
            # wrist joint + 0.2 * (wrist joint - elbow joint) ~ center of hand. Then, sample a small square region
            # around it.
            # height_slice = slice(int(distal_joint[1] - sample_size), int(distal_joint[1] + sample_size))
            # width_slice = slice(int(distal_joint[0] - sample_size), int(distal_joint[0] + sample_size))
            # cs_slice = slice(None, None)
            sample = frame_cs[
                     int(distal_joint[1] - sample_size):int(distal_joint[1] + sample_size),
                     int(distal_joint[0] - sample_size):int(distal_joint[0] + sample_size),
                     :,
                     ]
            sample = np.mean(np.mean(sample, axis=0), axis=0)
            lower = sample - np.array([7, 60, 200])
            upper = sample + np.array([7, 60, 200])
            skin_mask = cv.inRange(frame_cs, lower, upper)
            # apply a series of erosions and dilations to the mask using an elliptical kernel
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(3, 3))
            skin_mask_eroded = cv.erode(skin_mask, kernel, iterations=2)
            skin_mask_dilated = cv.dilate(skin_mask_eroded, kernel, iterations=2)
            mask = np.multiply(skel_mask, skin_mask_dilated)
            # Find the connected component that contain the ankle
            comps, labels = cv.connectedComponents(mask)
            this_comp_label = labels[np.uint(distal_joint[1]), np.uint(distal_joint[0])]
            if this_comp_label == 0:  # use all
                mask = mask
            else:
                mask = np.uint8(labels == this_comp_label)

            if mask.any():
                mask = np.uint8(mask / np.max(mask[:]))
            masks[limb_c] = mask
        # post process
        bkg_flow = get_bkg_flow_vector(skel, flow)
        for limb_c, limb in enumerate(limbs):
            failed_ret = np.array([[], []])
            mask = masks[limb_c]
            if mask is None:
                flow_feature_per_limb[limb_c] = failed_ret
                continue

            if flow_feature_per_limb[limb_c] == failed_ret:
                continue

            fx, fy = np.where(mask > 0)  # This is numpy, so agree with outside

            if fx.size < 10 or fy.size < 10:
                flow_feature_per_limb[limb_c] = failed_ret
                continue

            # check if they are close too borders
            if np.min(fx) < 10 or np.min(fy) < 10 or np.min(fx) > frame.shape[0] - 10 or np.min(fy) > frame.shape[1] - 10:
                flow_feature_per_limb[limb_c] = failed_ret
                continue

            flows = flow[fx, fy, ::-1]  # This is from opencv, so we need to swap x and y
            # Compare to bkg flow
            dist_to_bkg = np.linalg.norm(flows - np.tile(bkg_flow, [flows.shape[0], 1]), axis=1) / ref_len
            good_inds = np.where(dist_to_bkg > 0.2 / 100.0)[0]
            if good_inds.size < 20:  # the area seems does not move at all, it cost nothing to just include all
                good_inds = range(flows.shape[0])
            good_flows = flows[good_inds, :]
            good_fx = fx[good_inds]
            good_fy = fy[good_inds]

            good_coords = np.array([good_fx, good_fy]).transpose()  # This is numpy, so agree with outside
            good_mask = np.zeros(flow.shape[:2], dtype='uint8')
            good_mask[good_fx, good_fy] = 1
            if t == 0 or distal_centroids[t - 1, limb_c, 0] == -1:  # prev bad
                distal_centroids[t, limb_c, :2] = np.mean(good_coords, axis=0)
                distal_centroids[t, limb_c, 2] = skel[limb[1], 2]
            else:
                distal_centroids[t, limb_c, :2] = distal_centroids[t - 1, limb_c, :2] + np.mean(good_flows, axis=0)
                distal_centroids[t, limb_c, 2] = skel[limb[1], 2]

            flow_feature_per_limb[limb_c] = good_coords, good_flows
        flow_features.append(flow_feature_per_limb)
    return flow_features, distal_centroids


def get_bkg_flow_vector(skel, flow):
    skel = skel[skel[:, 2] > JOINT_SCORE_THRESH]
    skel1 = skel[:, 0:2]
    if skel1.shape[0] == 0:
        minx = 1
        maxx = 2
        miny = 1
        maxy = 2
    else:
        minx = int(max(np.min(skel1[:, 0]) - 20, 0))
        maxx = int(min(np.max(skel1[:, 0]) + 20, flow.shape[0]))
        miny = int(max(np.min(skel1[:, 1]) - 20, 0))
        maxy = int(min(np.max(skel1[:, 1]) + 20, flow.shape[1]))
    bkg_mask = np.ones(flow.shape[0:2], dtype='uint8')
    bkg_mask[minx:maxx, miny:maxy] = 0
    flow_bkg = np.multiply(flow, np.dstack([bkg_mask, bkg_mask]))
    flow_bkg_vector = np.sum(np.sum(flow_bkg, axis=0), axis=0) / np.sum(bkg_mask[:])
    return flow_bkg_vector


def get_background_flow(smoothed_skeletons, video_filepath):
    """Get average flow for the image background. Use skeleton to infer foreground."""
    cap = cv.VideoCapture(video_filepath)
    num_frames = smoothed_skeletons.shape[0]
    avg_flow_bkg = np.zeros(num_frames - 1)
    _, previous_frame = cap.read()
    for t in range(num_frames - 1):
        prev_frame_grey = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
        _, frame = cap.read()
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_frame_grey, frame_grey, None,
                                           pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                           poly_n=5, poly_sigma=1.2, flags=0)
        previous_frame = frame
        smoothed_skeleton = smoothed_skeletons[t]
        smoothed_skeleton = smoothed_skeleton[smoothed_skeleton[:, 2] > JOINT_SCORE_THRESH]
        skel1 = smoothed_skeleton[:, :2]
        if skel1.shape[0] == 0:
            minx = 1
            maxx = 2
            miny = 1
            maxy = 2
        else:
            minx = int(max(np.min(skel1[:, 0]) - 20, 0))
            maxx = int(min(np.max(skel1[:, 0]) + 20, flow.shape[0]))
            miny = int(max(np.min(skel1[:, 1]) - 20, 0))
            maxy = int(min(np.max(skel1[:, 1]) + 20, flow.shape[1]))
        flow_bkg = np.sum(np.linalg.norm(flow, axis=2)) - np.sum(np.linalg.norm(flow[minx:maxx, miny:maxy, :], axis=2))
        area_bkg = flow.shape[0] * flow.shape[1] - (maxx - minx) * (maxy - miny)
        avg_flow_bkg[t] = flow_bkg / area_bkg
    return avg_flow_bkg


def get_scoreable_windows_wrt_flow(background_flow, background_flow_thresh,
                                   start_frame, window_length, window_stride) -> np.ndarray:
    win_starts = range(start_frame, background_flow.size - window_length, window_stride)
    num_windows = len(win_starts)
    scorable_wins_flow = np.ones(num_windows)
    for window_index, window_start_frame in enumerate(win_starts):
        win_perc = np.percentile(background_flow[window_start_frame:window_start_frame + window_length], q=75)
        if win_perc > background_flow_thresh:
            scorable_wins_flow[window_index] = 0
    return scorable_wins_flow
