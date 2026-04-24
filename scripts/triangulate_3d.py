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

# ---------------------------------------------------------------------------
# Hartley-Sturm optimal triangulation (degree-6 polynomial method)
# Reference: Hartley & Zisserman, "Multiple View Geometry", Algorithm 12.1
# ---------------------------------------------------------------------------

def _skew(v):
    """3x3 skew-symmetric matrix for cross product."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ], dtype=np.float64)


def _compute_fundamental_matrix(P1, P2):
    """F = [e2]_x @ P2 @ pinv(P1), where e2 = P2 @ null(P1)."""
    _, _, Vt = np.linalg.svd(P1)
    C1 = Vt[-1]                          # 4D homogeneous camera centre
    e2 = P2 @ C1                         # epipole in image 2
    return _skew(e2) @ P2 @ np.linalg.pinv(P1)


def _optimal_correct(x1, x2, F):
    """
    Hartley-Sturm optimal correction.
    Finds corrected 2D points x1_corr, x2_corr (pixel coords, shape (2,))
    that satisfy the epipolar constraint x2'^T F x1' = 0 and minimise
    the total squared reprojection error.

    Solves the degree-6 polynomial  g(t) = t·Q(t)² − (ad−bc)·P(t)²·L(t)·M(t) = 0
    where the polynomial elements come from the transformed fundamental matrix.
    """
    EPS = 1e-12

    # Step 1: translate image coords so x1, x2 are at origin
    T1 = np.array([[1, 0, -x1[0]], [0, 1, -x1[1]], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, -x2[0]], [0, 1, -x2[1]], [0, 0, 1]], dtype=np.float64)

    F_hat = np.linalg.inv(T2).T @ F @ np.linalg.inv(T1)

    # Step 2: epipoles of F_hat — normalise so that ||e[:2]|| = 1
    _, _, Vt = np.linalg.svd(F_hat)
    e1 = Vt[-1].copy()
    n1 = np.linalg.norm(e1[:2])
    if n1 < EPS:               # epipole at infinity → skip correction
        return x1, x2
    e1 /= n1

    _, _, Vt_T = np.linalg.svd(F_hat.T)
    e2_ep = Vt_T[-1].copy()
    n2 = np.linalg.norm(e2_ep[:2])
    if n2 < EPS:
        return x1, x2
    e2_ep /= n2

    # Step 3: rotation matrices that send epipoles to (1, 0, ...)
    def _rot(e):
        return np.array([
            [ e[0],  e[1], 0],
            [-e[1],  e[0], 0],
            [    0,     0, 1],
        ], dtype=np.float64)

    R1 = _rot(e1)
    R2 = _rot(e2_ep)

    f1 = e1[2]          # z-component (focal length of epipole in image 1)
    f2 = e2_ep[2]       # z-component in image 2

    # Step 4: transform fundamental matrix to canonical form
    F_tilde = R2 @ F_hat @ R1.T

    # Step 5: polynomial coefficients from top-left 2x2 of F_tilde
    a = F_tilde[0, 0]
    b = F_tilde[0, 1]
    c = F_tilde[1, 0]
    d = F_tilde[1, 1]

    # Step 6: build degree-6 polynomial g(t) = t·Q² − (ad−bc)·P²·L·M
    #   L(t) = a·t + b  [degree 1]
    #   M(t) = c·t + d  [degree 1]
    #   Q(t) = L² + f2²·M²  [degree 2]
    #   P(t) = 1 + f1²·t²   [degree 2]
    #
    # Coefficients stored highest-degree-first (numpy convention).
    L  = np.array([a, b])
    M  = np.array([c, d])
    Q  = np.polymul(L, L) + f2**2 * np.polymul(M, M)   # degree 2
    P_ = np.array([f1**2, 0.0, 1.0])                    # degree 2

    # t · Q²: np.polymul(Q,Q) is degree 4 (5 coeffs);
    # append 0 = ×t → degree 5 (6 coeffs); prepend 0 to pad to degree 6 (7 coeffs)
    tQ2 = np.concatenate([[0.0], np.append(np.polymul(Q, Q), 0.0)])

    # (ad−bc)·P²·L·M : degree 6
    rhs = (a * d - b * c) * np.polymul(np.polymul(P_, P_), np.polymul(L, M))

    poly = tQ2 - rhs  # degree 6 (7 coefficients)

    # Step 7: find all real roots
    roots = np.roots(poly)
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-6])

    # Step 8: evaluate cost C(t) = t²/(1+f1²t²) + (ct+d)²/((at+b)²+f2²(ct+d)²)
    def _cost(t):
        return (t**2 / (1.0 + f1**2 * t**2 + EPS) +
                (c*t + d)**2 / ((a*t + b)**2 + f2**2 * (c*t + d)**2 + EPS))

    # Also consider t → ∞ (cost = 1/f1² + c²/(a²+f2²c²))
    cost_inf = 1.0 / (f1**2 + EPS) + c**2 / (a**2 + f2**2 * c**2 + EPS)
    best_t, best_cost = None, cost_inf

    for t in real_roots:
        ct = float(t)
        cv = _cost(ct)
        if cv < best_cost:
            best_cost = cv
            best_t = ct

    if best_t is None:
        return x1, x2   # fallback: no better root found

    t = best_t

    # Step 9: corrected lines in canonical space
    l1_c = np.array([t * f1,          1.0,     -t])
    l2_c = np.array([-f2 * (c*t + d), a*t + b,  c*t + d])

    # Step 10: closest points on those lines to the origin
    def _foot(l):
        denom = l[0]**2 + l[1]**2
        if abs(denom) < EPS:
            return np.array([0.0, 0.0])
        return np.array([-l[0]*l[2] / denom, -l[1]*l[2] / denom])

    x1h = np.array([*_foot(l1_c), 1.0])
    x2h = np.array([*_foot(l2_c), 1.0])

    # Step 11: back-transform  →  pixel coords
    T1_inv = np.array([[1, 0, x1[0]], [0, 1, x1[1]], [0, 0, 1]], dtype=np.float64)
    T2_inv = np.array([[1, 0, x2[0]], [0, 1, x2[1]], [0, 0, 1]], dtype=np.float64)

    p1c = T1_inv @ (R1.T @ x1h)
    p2c = T2_inv @ (R2.T @ x2h)

    return p1c[:2] / p1c[2], p2c[:2] / p2c[2]


# Precompute fundamental matrix once
F_mat = _compute_fundamental_matrix(P1, P2)

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
                if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                    output_row.extend([0.0, 0.0, 0.0])
                    continue

                # Optimal correction (Hartley-Sturm degree-6 polynomial)
                x1_corr, x2_corr = _optimal_correct(
                    np.array([x1, y1]), np.array([x2, y2]), F_mat
                )

                # Undistort corrected points
                pts1_undist = cv2.undistortPoints(
                    x1_corr.reshape(1, 1, 2).astype(np.float32), mtx1, dist1, P=mtx1)
                pts2_undist = cv2.undistortPoints(
                    x2_corr.reshape(1, 1, 2).astype(np.float32), mtx2, dist2, P=mtx2)

                # Triangulate from corrected points
                points_4d = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)
                points_3d = points_4d[:3] / points_4d[3]  # Convert to 3D

                x3d, y3d, z3d = points_3d.flatten()
                output_row.extend([x3d, y3d, z3d])

        writer.writerow(output_row)

print("3D coordinates saved to {}".format(output_3d_csv))