import cv2
import numpy as np
import json
import argparse
import os

def approx_intrinsics(img_size):
    """Approximate camera intrinsics from image size."""
    h, w = img_size
    f = max(w, h)
    cx, cy = w / 2, h / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return K

def collect_feature_points(cam1_idx, cam2_idx, num_frames):
    """Collect SIFT feature points from live camera feeds."""
    cap1 = cv2.VideoCapture(cam1_idx)
    cap2 = cv2.VideoCapture(cam2_idx)

    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Cannot open cameras")

    sift = cv2.SIFT_create()
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    all_pts1 = []
    all_pts2 = []
    img_size = None

    print(f"Capturing {num_frames} frames...")

    for i in range(num_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print(f"Failed to capture frame {i}")
            continue

        if img_size is None:
            img_size = (frame1.shape[0], frame1.shape[1])

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Insufficient features in frame {i}")
            continue

        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            print(f"Insufficient matches in frame {i}: {len(good_matches)}")
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        all_pts1.extend(pts1)
        all_pts2.extend(pts2)

        print(f"Frame {i+1}/{num_frames}: {len(good_matches)} matches")

    cap1.release()
    cap2.release()

    if len(all_pts1) < 8:
        raise RuntimeError("Insufficient total matches for calibration")

    return np.array(all_pts1), np.array(all_pts2), img_size

def calibrate_from_features(pts1, pts2, img_size):
    """Compute calibration parameters from feature points."""
    # Approximate intrinsics
    K = approx_intrinsics(img_size)
    dist = np.zeros(5, dtype=np.float32)

    # Find fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99)
    if F is None:
        raise RuntimeError("Failed to find fundamental matrix")

    # Find essential matrix
    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("Failed to find essential matrix")

    # Recover pose
    retval, R, T, mask_p = cv2.recoverPose(E, pts1, pts2, K)
    if retval < 8:
        raise RuntimeError("Failed to recover pose")

    return K, dist, K, dist, R, T

def save_calibration(mtx1, dist1, mtx2, dist2, R, T, output_dir):
    """Save calibration parameters to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    for i, (mtx, dist) in enumerate([(mtx1, dist1), (mtx2, dist2)], 1):
        path = os.path.join(output_dir, f"camera_params{i}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_matrix": mtx.tolist(),
                "dist_coefficients": dist.tolist(),
                "status": "feature_calibrated",
                "method": "SIFT + Fundamental Matrix"
            }, f, indent=4)
        print(f"Saved: {path}")

    stereo_path = os.path.join(output_dir, "matrix_vector.json")
    with open(stereo_path, "w", encoding="utf-8") as f:
        json.dump({
            "rotation_matrix": R.tolist(),
            "vector": T.tolist()
        }, f, indent=4)
    print(f"Saved: {stereo_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature-based self-calibration from live cameras"
    )
    parser.add_argument("--cam1", type=int, default=0, help="Camera 1 index (default: 0)")
    parser.add_argument("--cam2", type=int, default=1, help="Camera 2 index (default: 1)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to capture (default: 60)")
    args = parser.parse_args()

    try:
        pts1, pts2, img_size = collect_feature_points(args.cam1, args.cam2, args.frames)
        print(f"Total points: {len(pts1)}")

        mtx1, dist1, mtx2, dist2, R, T = calibrate_from_features(pts1, pts2, img_size)

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_data")
        save_calibration(mtx1, dist1, mtx2, dist2, R, T, output_dir)

        print("Calibration completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)