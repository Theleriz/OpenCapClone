import cv2
import numpy as np
import os
import json

# --- Настройки ---
CHECKERBOARD = (10, 8)
SQUARE_SIZE = 25

# Пути к папкам с изображениями для каждой камеры
CAM1_IMAGES_PATH = r'C:\projects\ProjectX\cam1_imgs'
CAM2_IMAGES_PATH = r'C:\projects\ProjectX\cam2_imgs'


def calibrate_camera(images_dir):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    images = [
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ]

    if not images:
        raise FileNotFoundError(f"No images found in: {os.path.abspath(images_dir)}")

    print(f"Found {len(images)} images in {os.path.abspath(images_dir)}")

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  ✗ Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                 cv2.CALIB_CB_FAST_CHECK)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(100)
            print(f"  ✓ {os.path.basename(fname)}")
        else:
            print(f"  ✗ {os.path.basename(fname)} — corners not found")

    cv2.destroyAllWindows()

    if not objpoints:
        raise RuntimeError(
            "No chessboard corners detected in any image.\n"
            f"  • Check CHECKERBOARD = {CHECKERBOARD} matches your physical board's INNER corners\n"
            f"  • Inner corners = (columns - 1, rows - 1) = (10, 8) for an 11×9 board\n"
            "  • Make sure images are well-lit and the full board is visible"
        )

    print(f"Calibrating with {len(objpoints)} valid images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(f"Reprojection error: {ret:.4f} (good if < 0.5)")
    return mtx, dist


def save_params(mtx, dist, output_dir, filename):
    data_to_save = {
        "camera_matrix": mtx.tolist(),
        "dist_coefficients": dist.tolist(),
        "status": "calibrated",
        "checkerboard": CHECKERBOARD
    }

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"--- Данные успешно сохранены в {file_path} ---\n")


if __name__ == "__main__":
    output_dir = "output_data"

    print("=== Калибровка камеры 1 ===")
    mtx1, dist1 = calibrate_camera(CAM1_IMAGES_PATH)
    save_params(mtx1, dist1, output_dir, "camera_params1.json")

    print("=== Калибровка камеры 2 ===")
    mtx2, dist2 = calibrate_camera(CAM2_IMAGES_PATH)
    save_params(mtx2, dist2, output_dir, "camera_params2.json")