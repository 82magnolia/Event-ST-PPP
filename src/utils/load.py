import numpy as np
import pandas as pd
import cv2

def undo_distortion(src, instrinsic_matrix, distco=None):
    dst = cv2.undistortPoints(src, instrinsic_matrix, distco, None, instrinsic_matrix)
    return dst


def do_distortion(src, instrinsic_matrix, distco=None):
    x = src[..., 0]
    y = src[..., 1]
    k1 = distco[0]
    fx = instrinsic_matrix[0, 0]
    fy = instrinsic_matrix[1, 1]
    cx = instrinsic_matrix[0, 2]
    cy = instrinsic_matrix[1, 2]

    norm_x = (x - cx) / fx
    norm_y = (y - cy) / fy

    r2 = norm_x * norm_x + norm_y * norm_y

    distort_x = norm_x * (1 + k1 * r2)
    distort_y = norm_y * (1 + k1 * r2)
    
    distort_x = distort_x * fx + cx
    distort_y = distort_y * fy + cy

    distort_pts = np.stack([distort_x, distort_y], axis=-1)
    return distort_pts


def load_dataset(name, path_dataset, sequence, return_intrinsics=False):
    if name == "DAVIS_240C":
        calib_data = np.loadtxt('{}/{}/calib.txt'.format(path_dataset,sequence))
        events = pd.read_csv(
            '{}/{}/events.txt'.format(path_dataset,sequence), sep=" ", header=None)
        events.columns = ["ts", "x", "y", "p"]

        # Load intrinsics and distortion parameters from COLMAP reconstructions
        skip_rows = 3
        cam_labels_name = f"{path_dataset}/{sequence}/sparse/cameras.txt"
        raw_cam_labels = open(cam_labels_name, 'r').readlines()
        values = raw_cam_labels[skip_rows].strip().split(' ')
        instrinsic_matrix = np.zeros([3, 3])
        fx = float(values[4])
        fy = float(values[4])
        px = float(values[5])
        py = float(values[6])
        instrinsic_matrix[0, 0] = values[4]
        instrinsic_matrix[1, 1] = values[4]
        instrinsic_matrix[0, 2] = values[5]
        instrinsic_matrix[1, 2] = values[6]
        instrinsic_matrix[2, 2] = 1
        dist_co = np.array([float(values[7]), 0., 0., 0.])
        height = 180
        width = 240

        LUT = np.zeros([width, height, 2])
        for i in range(width):
            for j in range(height):
                LUT[i][j] = np.array([i, j])
        LUT = LUT.reshape((-1, 1, 2))
        LUT = undo_distortion(LUT, instrinsic_matrix, dist_co).reshape((width, height, 2))
        events_set = events.to_numpy()
        events_set[:, 0] -= events_set[0, 0]
    print("Events total count: ", len(events_set))
    print("Time duration of the sequence: {} s".format(events_set[-1][0] - events_set[0][0]))

    if return_intrinsics:
        return LUT, events_set, height, width, fx, fy, px, py, dist_co, instrinsic_matrix
    else:
        return LUT, events_set, height, width, fx, fy, px, py
