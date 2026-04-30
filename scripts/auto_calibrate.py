import cv2
import numpy as np
import os
import json
import argparse
import time

base_path = os.path.dirname(os.path.abspath(__file__))

CHECKERBOARD = (7, 7)
SQUARE_SIZE = 25  # mm

DETECT_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FAST_CHECK
)
REFINE_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)


def _make_objp():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def _side_by_side(frame1, frame2, target_w=1280):
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    scale = target_w / (w1 + w2)
    f1 = cv2.resize(frame1, (int(w1 * scale), int(h1 * scale)))
    f2 = cv2.resize(frame2, (int(w2 * scale), int(h2 * scale)))
    max_h = max(f1.shape[0], f2.shape[0])

    def pad(img):
        if img.shape[0] < max_h:
            p = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, p])
        return img

    return np.hstack([pad(f1), pad(f2)])


def collect_frames(cam1_idx, cam2_idx, target_frames, min_interval_sec, min_movement_px):
    objp = _make_objp()
    objpoints, imgpoints1, imgpoints2 = [], [], []
    last_corners1 = None
    last_capture_time = 0.0
    img_size = None
    flash_until = 0.0

    cap1 = cv2.VideoCapture(cam1_idx)
    cap2 = cv2.VideoCapture(cam2_idx)
    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        raise RuntimeError(f"Could not open cameras {cam1_idx} and/or {cam2_idx}")

    print(f"Cameras {cam1_idx} and {cam2_idx} opened.")
    print(f"Target: {target_frames} paired frames. Move the board to different positions/angles.")
    print("Press 'q' to abort early.\n")

    while len(objpoints) < target_frames:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Camera read failed.")
            break

        if img_size is None:
            img_size = (frame1.shape[1], frame1.shape[0])

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        found1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, DETECT_FLAGS)
        found2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, DETECT_FLAGS)

        now = time.time()

        if found1 and found2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), REFINE_CRITERIA)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), REFINE_CRITERIA)

            time_ok = (now - last_capture_time) >= min_interval_sec
            if last_corners1 is not None:
                movement = float(np.mean(np.linalg.norm(
                    corners1.reshape(-1, 2) - last_corners1.reshape(-1, 2), axis=1
                )))
            else:
                movement = min_movement_px + 1.0

            if time_ok and movement >= min_movement_px:
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                last_corners1 = corners1.copy()
                last_capture_time = now
                flash_until = now + 0.4
                print(f"  Captured {len(objpoints)}/{target_frames}  (movement={movement:.1f}px)")

            cv2.drawChessboardCorners(frame1, CHECKERBOARD, corners1, True)
            cv2.drawChessboardCorners(frame2, CHECKERBOARD, corners2, True)

        # Green flash border on capture
        if now < flash_until:
            for frame in [frame1, frame2]:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 255, 0), 12)
            cv2.putText(frame1, "CAPTURED!", (10, frame1.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

        # Status overlay
        count_str = f"Frames: {len(objpoints)}/{target_frames}"
        if found1 and found2:
            board_str, board_color = "BOARD: BOTH OK", (0, 255, 0)
        elif found1:
            board_str, board_color = "LEFT OK / RIGHT MISSING", (0, 165, 255)
        elif found2:
            board_str, board_color = "LEFT MISSING / RIGHT OK", (0, 165, 255)
        else:
            board_str, board_color = "NO BOARD DETECTED", (0, 0, 255)

        for frame in [frame1, frame2]:
            cv2.putText(frame, count_str, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, board_str, (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, board_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "q = quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Auto Calibration  (CAM1 | CAM2)", _side_by_side(frame1, frame2))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Aborted by user.")
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    return objpoints, imgpoints1, imgpoints2, img_size


def run_calibration(objpoints, imgpoints1, imgpoints2, img_size):
    print(f"\nCalibrating individual cameras on {len(objpoints)} frames...")

    ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints1, img_size, None, None)
    print(f"  Cam1 reprojection error: {ret1:.4f}  {'OK' if ret1 < 0.5 else 'HIGH — consider recalibrating'}")

    ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpoints2, img_size, None, None)
    print(f"  Cam2 reprojection error: {ret2:.4f}  {'OK' if ret2 < 0.5 else 'HIGH — consider recalibrating'}")

    print("Running stereo calibration...")
    ret_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2,
        img_size,
        criteria=STEREO_CRITERIA,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"  Stereo reprojection error: {ret_s:.4f}  {'OK' if ret_s < 1.0 else 'HIGH — consider recalibrating'}")

    return mtx1, dist1, mtx2, dist2, R, T


def save_all(mtx1, dist1, mtx2, dist2, R, T, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, (mtx, dist) in enumerate([(mtx1, dist1), (mtx2, dist2)], 1):
        path = os.path.join(output_dir, f"camera_params{i}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "camera_matrix": mtx.tolist(),
                "dist_coefficients": dist.tolist(),
                "status": "calibrated",
                "checkerboard": CHECKERBOARD
            }, f, indent=4)
        print(f"  Saved: {path}")

    stereo_path = os.path.join(output_dir, "matrix_vector.json")
    with open(stereo_path, "w", encoding="utf-8") as f:
        json.dump({
            "rotation_matrix": R.tolist(),
            "vector": T.tolist()
        }, f, indent=4)
    print(f"  Saved: {stereo_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto stereo calibration from live cameras"
    )
    parser.add_argument("--cam1",    type=int,   default=0,    help="Camera 1 index (default: 0)")
    parser.add_argument("--cam2",    type=int,   default=1,    help="Camera 2 index (default: 1)")
    parser.add_argument("--frames",  type=int,   default=30,   help="Target paired frames (default: 30)")
    parser.add_argument("--interval",type=float, default=1.5,  help="Min seconds between captures (default: 1.5)")
    parser.add_argument("--movement",type=float, default=25.0, help="Min avg corner movement in px (default: 25)")
    args = parser.parse_args()

    objpoints, imgpoints1, imgpoints2, img_size = collect_frames(
        args.cam1, args.cam2, args.frames, args.interval, args.movement
    )

    if len(objpoints) < 10:
        print(f"\nOnly {len(objpoints)} frames collected — need at least 10. Aborting.")
        raise SystemExit(1)

    mtx1, dist1, mtx2, dist2, R, T = run_calibration(objpoints, imgpoints1, imgpoints2, img_size)

    out_dir = os.path.join(base_path, "..", "output_data")
    print("\nSaving results...")
    save_all(mtx1, dist1, mtx2, dist2, R, T, out_dir)

    print("\nAuto calibration complete! All params saved to output_data/")
