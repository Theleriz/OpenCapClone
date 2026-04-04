import cv2
import numpy as np
import json
import csv
import os

# --- Paths ---
base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_path, '..', 'output_data')
cam1_csv = os.path.join(output_dir, 'hands_coords_cam1.csv')
cam2_csv = os.path.join(output_dir, 'hands_coords_cam2.csv')  # Assuming cam3 is cam2
cam1_params = os.path.join(output_dir, 'camera_params1.json')
cam2_params = os.path.join(output_dir, 'camera_params2.json')
stereo_params = os.path.join(output_dir, 'matrix_vector.json')
output_3d_csv = os.path.join(output_dir, 'hands_3d_coords.csv')

# --- Load camera parameters ---
def load_camera_params(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coefficients']).flatten()
    return mtx, dist

mtx1, dist1 = load_camera_params(cam1_params)
mtx2, dist2 = load_camera_params(cam2_params)

# --- Load stereo parameters ---
with open(stereo_params, 'r') as f:
    stereo_data = json.load(f)
R = np.array(stereo_data['rotation_matrix'])
T = np.array(stereo_data['vector']).flatten()

# --- Compute projection matrices ---
P1 = mtx1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = mtx2 @ np.hstack((R, T.reshape(3,1)))

# --- Load CSV data ---
def load_csv_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data), header

data1, header1 = load_csv_data(cam1_csv)
data2, header2 = load_csv_data(cam2_csv)

# Assume same number of frames
num_frames = min(len(data1), len(data2))

# Number of hands: 2, points per hand: 21
num_hands = 2
points_per_hand = 21

# Prepare output header
output_header = ['frame_idx', 'timestamp_ms']
for h in range(num_hands):
    for p in range(points_per_hand):
        output_header.extend(['h{}_p{}_x'.format(h, p), 'h{}_p{}_y'.format(h, p), 'h{}_p{}_z'.format(h, p)])

# Open output CSV
with open(output_3d_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(output_header)

    for frame_idx in range(num_frames):
        row1 = data1[frame_idx]
        row2 = data2[frame_idx]

        frame_num = int(row1[0])
        timestamp = int(row1[1])

        output_row = [frame_num, timestamp]

        for h in range(num_hands):
            for p in range(points_per_hand):
                # Get 2D points
                x1 = row1[2 + h*42 + p*2]
                y1 = row1[2 + h*42 + p*2 + 1]
                x2 = row2[2 + h*42 + p*2]
                y2 = row2[2 + h*42 + p*2 + 1]

                # Skip if no detection (0,0)
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    output_row.extend([0.0, 0.0, 0.0])
                    continue

                # Undistort points
                pts1_undist = cv2.undistortPoints(np.array([[x1, y1]], dtype=np.float32), mtx1, dist1, P=mtx1)
                pts2_undist = cv2.undistortPoints(np.array([[x2, y2]], dtype=np.float32), mtx2, dist2, P=mtx2)

                # Triangulate
                points_4d = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)
                points_3d = points_4d[:3] / points_4d[3]  # Convert to 3D

                x3d, y3d, z3d = points_3d.flatten()
                output_row.extend([x3d, y3d, z3d])

        writer.writerow(output_row)

print("3D coordinates saved to {}".format(output_3d_csv))