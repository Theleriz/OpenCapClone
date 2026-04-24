import cv2
import numpy as np
import os

# --- PATH SETTINGS ---
base_path  = os.path.dirname(os.path.abspath(__file__))
VIDEO1     = os.path.join(base_path, '..', 'media', 'calib01.mp4')
VIDEO2     = os.path.join(base_path, '..', 'media', 'calib02.mp4')
OUT_CAM1   = os.path.join(base_path, '..', 'cam1_imgs')
OUT_CAM2   = os.path.join(base_path, '..', 'cam2_imgs')

# --- CHECKERBOARD (inner corners, как в calibration.py) ---
CHECKERBOARD = (10, 8)

# Минимальный интервал между автосохранёнными кадрами (в кадрах видео)
AUTO_MIN_INTERVAL = 15

DETECT_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK)

REFINE_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_board(gray):
    """Возвращает (True, corners) или (False, None)."""
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, DETECT_FLAGS)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), REFINE_CRIT)
    return ret, corners


def draw_overlay(frame, found, corners, label, saved_count, auto_mode):
    out = frame.copy()
    color = (0, 220, 0) if found else (0, 0, 220)
    cv2.rectangle(out, (0, 0), (out.shape[1]-1, out.shape[0]-1), color, 6)
    if found and corners is not None:
        cv2.drawChessboardCorners(out, CHECKERBOARD, corners, True)
    status = "FOUND" if found else "NOT FOUND"
    cv2.putText(out, f"{label}: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, f"Saved: {saved_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mode_text = "[A] AUTO ON" if auto_mode else "[A] AUTO OFF"
    cv2.putText(out, mode_text, (10, out.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255) if auto_mode else (180, 180, 180), 1)
    return out


def make_side_by_side(f1, f2, target_width=1280):
    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]
    scale = target_width / (w1 + w2)
    f1r = cv2.resize(f1, (int(w1 * scale), int(h1 * scale)))
    f2r = cv2.resize(f2, (int(w2 * scale), int(h2 * scale)))
    # выравниваем по высоте
    max_h = max(f1r.shape[0], f2r.shape[0])
    def pad_h(img, h):
        if img.shape[0] < h:
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img
    return np.hstack([pad_h(f1r, max_h), pad_h(f2r, max_h)])


def save_pair(frame1, frame2, saved_count):
    fname = f"calib_{saved_count:04d}.jpg"
    cv2.imwrite(os.path.join(OUT_CAM1, fname), frame1)
    cv2.imwrite(os.path.join(OUT_CAM2, fname), frame2)
    print(f"  Saved pair #{saved_count}: {fname}")


if __name__ == "__main__":
    os.makedirs(OUT_CAM1, exist_ok=True)
    os.makedirs(OUT_CAM2, exist_ok=True)

    cap1 = cv2.VideoCapture(VIDEO1)
    cap2 = cv2.VideoCapture(VIDEO2)

    for path, cap in [(VIDEO1, cap1), (VIDEO2, cap2)]:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {path}")

    total = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap1.get(cv2.CAP_PROP_FPS)
    print(f"Videos opened. Frames: {total}, FPS: {fps:.1f}")
    print()
    print("Controls:")
    print("  SPACE  — save current frame pair (when board found in both)")
    print("  A      — toggle auto-save mode")
    print("  →      — skip 30 frames forward")
    print("  Q/Esc  — quit")
    print()

    saved_count   = 0
    frame_idx     = 0
    auto_mode     = False
    last_auto_save = -AUTO_MIN_INTERVAL

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("End of video.")
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        found1, corners1 = detect_board(gray1)
        found2, corners2 = detect_board(gray2)
        both_found = found1 and found2

        # Автосохранение
        if auto_mode and both_found and (frame_idx - last_auto_save) >= AUTO_MIN_INTERVAL:
            saved_count += 1
            save_pair(frame1, frame2, saved_count)
            last_auto_save = frame_idx

        ov1 = draw_overlay(frame1, found1, corners1, "CAM1", saved_count, auto_mode)
        ov2 = draw_overlay(frame2, found2, corners2, "CAM2", saved_count, auto_mode)
        canvas = make_side_by_side(ov1, ov2)

        # Подсказка по центру снизу
        hint = "SPACE: save" if both_found else "board not detected in both cameras"
        hint_color = (0, 255, 0) if both_found else (80, 80, 80)
        cv2.putText(canvas, hint,
                    (canvas.shape[1] // 2 - 160, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, hint_color, 2)

        cv2.putText(canvas, f"Frame {frame_idx}/{total}",
                    (canvas.shape[1] // 2 - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.imshow("Calibration frame extractor", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"Auto-save: {'ON' if auto_mode else 'OFF'}")
        elif key == ord(' '):
            if both_found:
                saved_count += 1
                save_pair(frame1, frame2, saved_count)
                last_auto_save = frame_idx
            else:
                print("  Skipped: board not found in both cameras")
        elif key == 83 or key == ord('d'):  # стрелка вправо
            for _ in range(29):
                cap1.read()
                cap2.read()
                frame_idx += 1

        frame_idx += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total pairs saved: {saved_count}")
    print(f"  cam1_imgs/: {OUT_CAM1}")
    print(f"  cam2_imgs/: {OUT_CAM2}")
