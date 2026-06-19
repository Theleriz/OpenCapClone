import cv2
import numpy as np
import os
import json

# --- Настройки ---
CHECKERBOARD = (7, 7)
SQUARE_SIZE  = 25  # mm

# Источники для каждой камеры — можно указать папку с фото ИЛИ путь к видео
base_path = os.path.dirname(os.path.abspath(__file__))
CAM1_SOURCE = os.path.join(base_path, '..', 'media', 'cal1l.mp4')
CAM2_SOURCE = os.path.join(base_path, '..', 'media', 'cal1r.mp4')


# Настройки для режима видео
VIDEO_SAMPLE_EVERY  = 5   # проверять каждый N-й кадр (остальные пропускать)
VIDEO_MIN_INTERVAL  = 15  # минимальный интервал между принятыми кадрами
VIDEO_MAX_FRAMES    = 40  # остановиться после сбора этого количества кадров


def _process_frame(img, objp, objpoints, imgpoints, criteria, flags, label):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        preview = img.copy()
        cv2.drawChessboardCorners(preview, CHECKERBOARD, corners2, True)
        cv2.imshow('Calibration', preview)
        cv2.waitKey(100)
        print(f"  ✓ {label}")
        return gray, True
    else:
        print(f"  ✗ {label} — corners not found")
        return gray, False


def calibrate_camera(source):
    """
    source — путь к папке с изображениями ИЛИ путь к видеофайлу (.mp4, .avi, ...).
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags    = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    gray      = None

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}
    is_video   = os.path.isfile(source) and os.path.splitext(source)[1].lower() in VIDEO_EXTS

    if is_video:
        # ── Режим видео ──────────────────────────────────────────────────────
        cap   = cv2.VideoCapture(source)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video source: {source}  ({total} frames @ {fps:.1f} fps)")
        print(f"Sampling every {VIDEO_SAMPLE_EVERY} frames, "
              f"min interval {VIDEO_MIN_INTERVAL}, target {VIDEO_MAX_FRAMES} frames")

        frame_idx      = 0
        last_accepted  = -VIDEO_MIN_INTERVAL

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            if frame_idx % VIDEO_SAMPLE_EVERY == 0:
                label = f"frame {frame_idx}/{total}"
                if (frame_idx - last_accepted) >= VIDEO_MIN_INTERVAL:
                    gray, found = _process_frame(
                        img, objp, objpoints, imgpoints, criteria, flags, label
                    )
                    if found:
                        last_accepted = frame_idx
                        if len(objpoints) >= VIDEO_MAX_FRAMES:
                            print(f"  Reached {VIDEO_MAX_FRAMES} frames, stopping early.")
                            break
                else:
                    # слишком близко к предыдущему принятому кадру — пропускаем
                    pass

            frame_idx += 1

        cap.release()

    else:
        # ── Режим папки с изображениями ──────────────────────────────────────
        images = sorted([
            os.path.join(source, f) for f in os.listdir(source)
            if f.lower().endswith(('.jpg', '.png', '.bmp'))
        ])
        if not images:
            raise FileNotFoundError(f"No images found in: {os.path.abspath(source)}")
        print(f"Image source: {os.path.abspath(source)}  ({len(images)} images)")

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"  ✗ Could not read: {fname}")
                continue
            gray, _ = _process_frame(
                img, objp, objpoints, imgpoints, criteria, flags, os.path.basename(fname)
            )

    cv2.destroyAllWindows()

    if not objpoints:
        raise RuntimeError(
            "No chessboard corners detected.\n"
            f"  • Check CHECKERBOARD = {CHECKERBOARD} matches your physical board's INNER corners\n"
            f"  • Inner corners = (columns-1, rows-1), e.g. (10, 8) for an 11×9 board\n"
            "  • Make sure the board is well-lit and fully visible"
        )

    print(f"Calibrating with {len(objpoints)} valid frames...")
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
    mtx1, dist1 = calibrate_camera(CAM1_SOURCE)
    save_params(mtx1, dist1, output_dir, "camera_params1.json")

    print("=== Калибровка камеры 2 ===")
    mtx2, dist2 = calibrate_camera(CAM2_SOURCE)
    save_params(mtx2, dist2, output_dir, "camera_params2.json")