import cv2
import numpy as np
import os
import json

base_path       = os.path.dirname(os.path.abspath(__file__))
output_data_dir = os.path.join(base_path, '..', 'output_data')

# --- Настройки ---
CHECKERBOARD = (7, 7)
SQUARE_SIZE  = 25  # мм

# Источники — папка с фото ИЛИ видеофайл (для обоих одинаково)
CAM1_SOURCE = os.path.join(base_path, '..', 'media', 'cal1l_synced.mp4')
CAM2_SOURCE = os.path.join(base_path, '..', 'media', 'cal1r_synced.mp4')

# Настройки для режима видео
VIDEO_SAMPLE_EVERY = 5   # проверять каждый N-й кадр
VIDEO_MIN_INTERVAL = 15  # минимальный интервал между принятыми парами
VIDEO_MAX_FRAMES   = 40  # остановиться после сбора этого количества пар

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}


def _load_camera_params(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mtx  = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['dist_coefficients'], dtype=np.float64).flatten()
    return mtx, dist


def _find_corners(gray, flags, refine_criteria):
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), refine_criteria)
    return ret, corners


def _show_pair(img1, img2, label, target_w=1280):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    scale  = target_w / (w1 + w2)
    f1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
    f2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
    max_h = max(f1.shape[0], f2.shape[0])
    def pad(img):
        if img.shape[0] < max_h:
            p = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, p])
        return img
    canvas = np.hstack([pad(f1), pad(f2)])
    cv2.putText(canvas, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.imshow('Stereo Calibration', canvas)
    cv2.waitKey(200)


def stereo_calibrate(source1, source2):
    """
    source1/source2 — путь к папке с изображениями ИЛИ к видеофайлу.
    Для видео оба файла читаются синхронно — кадр за кадром.
    """
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    detect_flags    = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                       cv2.CALIB_CB_NORMALIZE_IMAGE +
                       cv2.CALIB_CB_FAST_CHECK)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints  = []
    imgpoints1 = []
    imgpoints2 = []
    gray1 = gray2 = None

    is_video = (os.path.isfile(source1) and
                os.path.splitext(source1)[1].lower() in VIDEO_EXTS)

    if is_video:
        # ── Режим видео ──────────────────────────────────────────────────────
        cap1  = cv2.VideoCapture(source1)
        cap2  = cv2.VideoCapture(source2)
        total = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap1.get(cv2.CAP_PROP_FPS)
        print(f"Video sources: {os.path.basename(source1)} | {os.path.basename(source2)}")
        print(f"  {total} frames @ {fps:.1f} fps  |  "
              f"sample every {VIDEO_SAMPLE_EVERY}, min interval {VIDEO_MIN_INTERVAL}, "
              f"target {VIDEO_MAX_FRAMES} pairs")

        frame_idx     = 0
        last_accepted = -VIDEO_MIN_INTERVAL

        while cap1.isOpened() and cap2.isOpened():
            ret1, img1 = cap1.read()
            ret2, img2 = cap2.read()
            if not ret1 or not ret2:
                break

            if frame_idx % VIDEO_SAMPLE_EVERY == 0 and \
               (frame_idx - last_accepted) >= VIDEO_MIN_INTERVAL:

                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                found1, corners1 = _find_corners(gray1, detect_flags, refine_criteria)
                found2, corners2 = _find_corners(gray2, detect_flags, refine_criteria)

                label = f"frame {frame_idx}/{total}  pairs: {len(objpoints)}"
                if found1 and found2:
                    objpoints.append(objp)
                    imgpoints1.append(corners1)
                    imgpoints2.append(corners2)
                    last_accepted = frame_idx
                    vis1 = img1.copy()
                    vis2 = img2.copy()
                    cv2.drawChessboardCorners(vis1, CHECKERBOARD, corners1, True)
                    cv2.drawChessboardCorners(vis2, CHECKERBOARD, corners2, True)
                    _show_pair(vis1, vis2, f"✓ {label}")
                    print(f"  ✓ frame {frame_idx}")
                    if len(objpoints) >= VIDEO_MAX_FRAMES:
                        print(f"  Reached {VIDEO_MAX_FRAMES} pairs, stopping early.")
                        break
                else:
                    status = f"cam1={'OK' if found1 else 'FAIL'}, cam2={'OK' if found2 else 'FAIL'}"
                    print(f"  ✗ frame {frame_idx} — {status}")

            frame_idx += 1

        cap1.release()
        cap2.release()

    else:
        # ── Режим папки с изображениями ──────────────────────────────────────
        images1 = sorted([os.path.join(source1, f) for f in os.listdir(source1)
                          if f.lower().endswith(('.jpg', '.png', '.bmp'))])
        images2 = sorted([os.path.join(source2, f) for f in os.listdir(source2)
                          if f.lower().endswith(('.jpg', '.png', '.bmp'))])

        print(f"cam1: {len(images1)} images | cam2: {len(images2)} images")
        if not images1 or not images2:
            raise FileNotFoundError(
                f"No images found.\n  cam1: {source1}\n  cam2: {source2}"
            )

        for img1_path, img2_path in zip(images1, images2):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            if img1 is None or img2 is None:
                print(f"  ✗ Could not read pair: {os.path.basename(img1_path)}")
                continue

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            found1, corners1 = _find_corners(gray1, detect_flags, refine_criteria)
            found2, corners2 = _find_corners(gray2, detect_flags, refine_criteria)

            name = os.path.basename(img1_path)
            if found1 and found2:
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                vis1, vis2 = img1.copy(), img2.copy()
                cv2.drawChessboardCorners(vis1, CHECKERBOARD, corners1, True)
                cv2.drawChessboardCorners(vis2, CHECKERBOARD, corners2, True)
                _show_pair(vis1, vis2, f"✓ {name}  pairs: {len(objpoints)}")
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} — cam1={'OK' if found1 else 'FAIL'}, "
                      f"cam2={'OK' if found2 else 'FAIL'}")

    cv2.destroyAllWindows()

    print(f"\nPaired frames collected: {len(objpoints)}")
    if not objpoints:
        raise RuntimeError(
            "No paired frames found.\n"
            "  • CHECKERBOARD must match inner corners (columns-1, rows-1)\n"
            "  • Videos must be already synchronized\n"
            "  • Board must be fully visible in both cameras simultaneously"
        )

    mtx1, dist1 = _load_camera_params(os.path.join(output_data_dir, 'camera_params1.json'))
    mtx2, dist2 = _load_camera_params(os.path.join(output_data_dir, 'camera_params2.json'))

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2,
        gray1.shape[::-1],
        criteria=stereo_criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Stereo reprojection error: {ret:.4f} (good if < 1.0)")
    return R, T


if __name__ == "__main__":
    R, T = stereo_calibrate(CAM1_SOURCE, CAM2_SOURCE)

    data_to_save = {
        "rotation_matrix": R.tolist(),
        "vector": T.tolist(),
    }

    file_path = os.path.join(output_data_dir, "matrix_vector.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"\n--- Stereo params saved to {file_path} ---")