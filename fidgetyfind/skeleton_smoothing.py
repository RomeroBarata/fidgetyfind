import itertools as it
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal.windows import gaussian


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        else:
            return json.JSONEncoder.default(self, obj)


def smooth_skeletons(skeletons_dir: str, save_dir: Optional[str] = None) -> np.ndarray:
    """Given Directory Containing Skeletons Detected by OpenPose, Smooth the Skeletons."""
    video_id = os.path.basename(skeletons_dir)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        smoothed_skeleton_file = os.path.join(save_dir, f'{video_id}.npy')
        skip_video = os.path.exists(smoothed_skeleton_file)
        if skip_video:
            print(f'Already smoothed skeleton for video {video_id}. Skipping it.')
            smoothed_skeletons = np.load(smoothed_skeleton_file)
            return smoothed_skeletons
    skeleton_json_file = os.path.join(skeletons_dir, f'{video_id}.json')
    if os.path.exists(skeleton_json_file):
        skeletons = load_json(skeleton_json_file)
    else:
        skeletons_pkl_file = os.path.join(skeletons_dir, f'{video_id}.pkl')
        df_pkl = pd.read_pickle(skeletons_pkl_file)
        df = df_pkl.groupby(['video', 'frame']).apply(get_skel)
        try:
            skeletons = df_to_skel(df)
        except KeyError:
            print(f'Failed to process video {video_id}')
            return
        save_json(skeletons, skeleton_json_file)
    preprocessed_skeletons, preprocessed_scores = preprocess_skel(skeletons)
    smoothed_skeletons = _smooth_skeletons(preprocessed_skeletons)
    if save_dir is not None:
        np.save(smoothed_skeleton_file, arr=smoothed_skeletons)
        print(f'Smoothed skeletons for video {video_id} and saved them to {smoothed_skeleton_file}')
    return smoothed_skeletons


def load_json(fname):
    with open(fname, 'r',) as fd:
        obj = json.load(fd)
    return obj


def save_json(obj, fname, sort_keys=True, indent=4, separators=None):
    with open(fname, 'w') as fd:
        json.dump(obj, fd, indent=indent, sort_keys=sort_keys, cls=ComplexEncoder, separators=separators)


def get_skel(df):
    if len(list(it.chain(*df.limbs_subset))) > 0:
        peaks = df.peaks.iloc[0]
        parts_in_skel = df.limbs_subset.iloc[0]
        person_to_peak_mapping = [list(i[:-2]) for i in parts_in_skel]
        skel_idx = [[i] * (len(iskel) - 2) for i, iskel in enumerate(parts_in_skel)]
        idx_df = pd.DataFrame.from_dict({'peak_idx': list(it.chain(*person_to_peak_mapping)),
                                         'person_idx': list(it.chain(*skel_idx))})
        peaks_list = list(it.chain.from_iterable(peaks))
        x = [ipeak[0] for ipeak in peaks_list]
        y = [ipeak[1] for ipeak in peaks_list]
        c = [ipeak[2] for ipeak in peaks_list]
        peak_idx = [ipeak[3] for ipeak in peaks_list]
        kp_idx = list(it.chain.from_iterable([len(ipeak) * [i] for i, ipeak in enumerate(peaks)]))
        peak_df = pd.DataFrame.from_dict({'x': x, 'y': y, 'c': c, 'peak_idx': peak_idx, 'part_idx': kp_idx})
        kp_df = pd.merge(idx_df, peak_df, on='peak_idx', how='left').drop('peak_idx', axis=1)
        kp_df = kp_df.loc[~kp_df.c.isnull(), :]
    else:
        kp_df = pd.DataFrame()
    return kp_df


def df_to_skel(df):
    # keep person index with max number of keypoints per frame
    counts = df.groupby(['video', 'frame', 'person_idx'])['c'].count().reset_index()
    max_rows = counts.groupby(['video', 'frame'])['c'].idxmax().tolist()
    max_rows_df = counts.loc[max_rows, ['video', 'frame', 'person_idx']]
    max_rows_df['dum'] = 1
    df = pd.merge(df.reset_index(), max_rows_df, on=['video', 'frame', 'person_idx'], how='inner')

    df = df[['video', 'frame', 'x', 'y', 'c', 'part_idx']]
    frames = np.sort(np.unique(df.frame.to_numpy()))
    skeletons = []
    for frame in frames:
        fdf = df[df['frame'] == frame]
        x = fdf.x.to_numpy()
        y = fdf.y.to_numpy()
        c = fdf.c.to_numpy()
        part_idx = fdf.part_idx.to_numpy()
        skel = np.zeros([3, 18])
        for i, idx in enumerate(part_idx):
            x_i = x[i]
            skel[:, int(idx)] = np.array([y[i], x_i, c[i]]).T
        s_coco = {
            'image_id': '%03d.jpg' % (int(frame)),
            'category_id': 1,
            'score': np.nanmean(c).item(),
            'keypoints': skel.T.flatten().tolist(),
        }
        skeletons.append(s_coco)
    return skeletons


def preprocess_skel(skeletons):
    BAD_SCORE = -1000.0
    JOINT_SCORE_THRESH = 0.1
    SKEL_SCORE_THRESH = 0.0
    # Convert the skeletons to numpy array: num_frames x num_joints x 3
    preprocessed_skeletons = []
    preprocessed_scores = []
    for skeleton_item in skeletons:
        frame_id = int(skeleton_item['image_id'].split(".")[0])
        skel_score = skeleton_item['score']
        skeleton = skeleton_item['keypoints']

        this_skel = []
        num_joint = int(len(skeleton) / 3)
        for i in range(num_joint):
            this_skel.append(np.array([skeleton[3 * i], skeleton[3 * i + 1], skeleton[3 * i + 2]]))
        this_skel = np.array(this_skel)
        # Mid-hip is not automatically detected by OpenPose. We average right and left hips.
        this_skel = np.insert(this_skel, 8, (this_skel[8, :] + this_skel[11, :]) / 2.0, axis=0)
        # Check scores
        if skel_score < SKEL_SCORE_THRESH:
            this_skel = np.zeros(this_skel.shape)
            skel_score = BAD_SCORE
        # Check scores of each joint
        for i in range(this_skel.shape[0]):
            if this_skel[i, 2] < JOINT_SCORE_THRESH:
                this_skel[i, :] = np.zeros(this_skel[i, :].shape)
                this_skel[i, 2] = BAD_SCORE

        # We suppose one skeleton per frame, so if there is more than one in a frame, we choose to keep
        # the higher scored one
        if frame_id < len(preprocessed_skeletons):
            # Skim the weaker one
            if skel_score > preprocessed_scores[-1]:
                preprocessed_skeletons[-1] = this_skel
                preprocessed_scores[-1] = skel_score

        # if missing some intermediate, fill in until reached
        while frame_id > len(preprocessed_skeletons):
            preprocessed_skeletons.append(np.zeros(this_skel.shape))
            preprocessed_scores.append(BAD_SCORE)

        if frame_id == len(preprocessed_skeletons):  # normal case
            preprocessed_skeletons.append(this_skel)
            preprocessed_scores.append(skel_score)
    preprocessed_skeletons = np.array(preprocessed_skeletons)
    return preprocessed_skeletons, preprocessed_scores


def _smooth_skeletons(skeletons):
    JOINT_SCORE_THRESH = 0.1
    BAD_SCORE = -1000.0
    INTERPOLATED_SCORE = -1.0
    window = 5
    sigma = 2
    kernel = gaussian(window, sigma, True)
    failure_ratio_threshold = window / 3
    smoothed_skeletons = np.zeros(skeletons.shape)
    for joint in range(skeletons.shape[1]):
        scores = skeletons[:, joint, 2]
        scores[scores <= JOINT_SCORE_THRESH] = BAD_SCORE
        # Set scores to -1000 means that you will ignore it no matter what because your smoothed score will be neg
        # Set it 0.0 means that it will be gotten from its neighbors which is shady
        # The best is to interpolate first, using a proper interpolator, then ignore it
        padded_score = np.concatenate((np.zeros(int(window / 2)), scores, np.zeros(int(window / 2))))
        for channel in range(3):
            # Smooth along time for each channel independently
            # Do we smooth scores or only smooth x and y?
            # yes, because when we smooth, locs are counted in with its neighbor already
            sig = skeletons[:, joint, channel]
            new_sig = np.zeros(sig.shape)
            # INTERPOLATE TO TRY TO FILL THE GAP
            x = np.where(scores > 0)[0]
            if x.shape[0] >= 2:  # Just for interp to work, may not make sense in reality
                x_new = np.where(scores <= 0)[0]
                # check that interpolated places are in the range
                x_new_range = []
                for xn in x_new:
                    if xn > np.min(x) and xn < np.max(x):
                        x_new_range.append(xn)
                x_new = np.array(x_new_range)
                y = sig[x]
                f = interpolate.interp1d(x, y)
                y_missing = f(x_new)
                for i, m in enumerate(x_new):
                    if np.sum(padded_score[m:m + window] < 0) < failure_ratio_threshold:  # a reliable interp
                        sig[m] = y_missing[i] if channel < 2 else INTERPOLATED_SCORE

            padded_sig = np.concatenate((np.zeros(int(window / 2)), sig, np.zeros(int(window / 2))))
            # SMOOTH OUT (INTERPOLATED) SEQUENCE
            for t in range(sig.shape[0]):
                if channel != 2:  # signal, weight = kernel * score
                    weight = np.multiply(kernel, padded_score[t:t + window])
                else:  # scores, just use kernel
                    weight = kernel

                if scores[t] > 0.0:  # We refuse to fill the empty places because we already interpolated above
                    new_sig[t] = np.dot(padded_sig[t:t + window], weight) / np.sum(weight)
                else:
                    # This is the case where there is not a single signal in the neighbor to interpolate
                    # We should give up and mark the bad
                    new_sig[t] = 0 if channel < 2 else BAD_SCORE
            smoothed_skeletons[:, joint, channel] = new_sig
    return smoothed_skeletons
