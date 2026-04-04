import cv2
import numpy as np
import os
import json

# --- Параметры камер ---
mtx1 = np.array([
    [1097.5840828812597, 0.0,              1025.247734218068],
    [0.0,               984.0683528693361, 385.81068894476743],
    [0.0,               0.0,              1.0]
])
dist1 = np.array([-0.48659317148416364, 0.8148522049914797,
                  -0.012198322959003607, 0.041759706725023424, -0.5303064734026546])

mtx2 = np.array([
    [1381.2870410610528, 0.0,               609.2616368532443],
    [0.0,               1410.1934137649462, 380.89647173154975],
    [0.0,               0.0,               1.0]
])
dist2 = np.array([0.3277110794081981, -10.040639694630121,
                  -0.008637784469658266, -0.08391471654141584, 49.07735821474451])

# 11 columns × 9 rows → inner corners = (10, 8)
CHECKERBOARD = (10, 8)
SQUARE_SIZE = 25  # мм

base_path = os.path.dirname(os.path.abspath(__file__))
CAM1_IMAGES_PATH = os.path.join(base_path, '..', 'cam1_imgs')
CAM2_IMAGES_PATH = os.path.join(base_path, '..', 'cam2_imgs')


def stereo_calibrate(dir1, dir2):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints  = []
    imgpoints1 = []
    imgpoints2 = []

    images1 = sorted([os.path.join(dir1, f) for f in os.listdir(dir1)
                      if f.lower().endswith(('.jpg', '.png', '.bmp'))])
    images2 = sorted([os.path.join(dir2, f) for f in os.listdir(dir2)
                      if f.lower().endswith(('.jpg', '.png', '.bmp'))])

    print(f"cam1: {len(images1)} images | cam2: {len(images2)} images")

    if not images1 or not images2:
        raise FileNotFoundError(
            f"No images found.\n  cam1: {os.path.abspath(dir1)}\n  cam2: {os.path.abspath(dir2)}"
        )

    gray1, gray2 = None, None
    for img1_path, img2_path in zip(images1, images2):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"  ✗ Could not read: {img1_path} or {img2_path}")
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, flags)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, flags)

        name = os.path.basename(img1_path)
        if ret1 and ret2:
            refine = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), refine)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), refine)
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} — cam1={'OK' if ret1 else 'FAIL'}, cam2={'OK' if ret2 else 'FAIL'}")

    print(f"\nPaired frames detected: {len(objpoints)}")

    if not objpoints:
        raise RuntimeError(
            "No paired frames found. Possible reasons:\n"
            "  • CHECKERBOARD size wrong — must be inner corners (columns-1, rows-1)\n"
            "  • Images not shot simultaneously from both cameras\n"
            "  • Board not fully visible in both views"
        )

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2,
        gray1.shape[::-1],
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Stereo reprojection error: {ret:.4f} (good if < 1.0)")
    return R, T


if __name__ == "__main__":
    R, T = stereo_calibrate(CAM1_IMAGES_PATH, CAM2_IMAGES_PATH)

    data_to_save = {
        "rotation_matrix": R.tolist(),
        "vector": T.tolist(),
    }

    output_dir = os.path.join(base_path, '..', 'output_data')
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "matrix_vector.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"\n--- Stereo params saved to {file_path} ---")