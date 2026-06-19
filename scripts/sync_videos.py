import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- PATH SETTINGS ---
base_path = os.path.dirname(os.path.abspath(__file__))
VIDEO1    = os.path.join(base_path, '..', 'media', 'cal1l.mp4')
VIDEO2    = os.path.join(base_path, '..', 'media', 'cal1r.mp4')
OUT_CAM1  = os.path.join(base_path, '..', 'media', 'cal1l_synced.mp4')
OUT_CAM2  = os.path.join(base_path, '..', 'media', 'cal1r_synced.mp4')

# --- FLASH DETECTION SETTINGS ---
SCAN_SCALE       = 0.25   # downscale factor for scanning speed
FLASH_SIGMA      = 4.0    # how many σ above mean counts as flash onset
SHOW_PLOT        = True   # show brightness signal plot before preview


# ── Brightness signal ──────────────────────────────────────────────────────────

def compute_brightness(video_path, scale=SCAN_SCALE):
    cap = cv2.VideoCapture(video_path)
    signal = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        signal.append(gray.mean())
    cap.release()
    return np.array(signal)


def find_flash_frame(brightness, sigma=FLASH_SIGMA):
    """
    Находит кадр включения света.
    Ищет первый скачок яркости выше mean + sigma*std в производной сигнала.
    """
    diff = np.diff(brightness, prepend=brightness[0])
    threshold = diff.mean() + sigma * diff.std()
    candidates = np.where(diff > threshold)[0]
    if len(candidates) == 0:
        return None
    return int(candidates[0])


# ── Frame access ───────────────────────────────────────────────────────────────

def get_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, width, height


# ── Brightness plot ────────────────────────────────────────────────────────────

def show_brightness_plot(br1, br2, flash1, flash2, fps1, fps2):
    t1 = np.arange(len(br1)) / fps1
    t2 = np.arange(len(br2)) / fps2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
    fig.suptitle('Brightness signal — verify flash detection', fontsize=12)

    ax1.plot(t1, br1, color='#4FC3F7', linewidth=0.8, label='cam1')
    if flash1 is not None:
        ax1.axvline(flash1 / fps1, color='#FF5722', linewidth=2,
                    label=f'flash @ frame {flash1}')
    ax1.set_ylabel('Mean brightness')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t2, br2, color='#81C784', linewidth=0.8, label='cam2')
    if flash2 is not None:
        ax2.axvline(flash2 / fps2, color='#FF5722', linewidth=2,
                    label=f'flash @ frame {flash2}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mean brightness')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Interactive preview ────────────────────────────────────────────────────────

def make_canvas(frame1, frame2, idx1, idx2, fps1, fps2, br1, br2, target_w=1280):
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    scale  = target_w / (w1 + w2)

    f1r = cv2.resize(frame1, (int(w1 * scale), int(h1 * scale)))
    f2r = cv2.resize(frame2, (int(w2 * scale), int(h2 * scale)))

    max_h = max(f1r.shape[0], f2r.shape[0])

    def pad(img):
        if img.shape[0] < max_h:
            p = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, p])
        return img

    canvas = np.hstack([pad(f1r), pad(f2r)])

    t1_ms  = (idx1 / fps1) * 1000
    t2_ms  = (idx2 / fps2) * 1000
    offset = t1_ms - t2_ms

    brightness_bar_h = 60
    bar = np.zeros((brightness_bar_h, canvas.shape[1], 3), dtype=np.uint8)

    info = [
        f"CAM1  frame={idx1}  t={t1_ms:.1f}ms  brightness={br1[idx1]:.1f}",
        f"CAM2  frame={idx2}  t={t2_ms:.1f}ms  brightness={br2[idx2]:.1f}",
        f"Offset: {offset:+.1f} ms   |   "
        f"LEFT/RIGHT: cam1 ±1     UP/DOWN: cam2 ±1     "
        f"SHIFT+arrows: ±10     ENTER: save     ESC: cancel",
    ]

    colors = [(0, 210, 255), (100, 230, 100), (200, 200, 200)]
    for i, (text, color) in enumerate(zip(info, colors)):
        cv2.putText(bar, text, (10, 18 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # flash marker overlay on each half
    mid = f1r.shape[1]
    cv2.rectangle(canvas, (0, 0), (mid - 1, 28), (0, 0, 0), -1)
    cv2.putText(canvas, f"CAM1  frame {idx1}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (mid, 0), (canvas.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(canvas, f"CAM2  frame {idx2}", (mid + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 230, 100), 2, cv2.LINE_AA)

    return np.vstack([canvas, bar])


def interactive_preview(video1, video2, flash1, flash2, br1, br2):
    fps1, total1, _, _ = get_video_info(video1)
    fps2, total2, _, _ = get_video_info(video2)

    idx1 = flash1
    idx2 = flash2

    WIN = "Sync preview"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    need_refresh = True
    frame1 = frame2 = None

    while True:
        if need_refresh:
            frame1 = get_frame(video1, idx1)
            frame2 = get_frame(video2, idx2)
            if frame1 is None or frame2 is None:
                print("Cannot read frame.")
                break
            canvas = make_canvas(frame1, frame2, idx1, idx2, fps1, fps2, br1, br2)
            cv2.imshow(WIN, canvas)
            need_refresh = False

        key = cv2.waitKey(30) & 0xFF

        if key == 13 or key == ord('\r'):   # ENTER — confirm
            cv2.destroyAllWindows()
            return idx1, idx2

        elif key == 27:                      # ESC — cancel
            cv2.destroyAllWindows()
            return None, None

        # Shift held = ±10 frames, otherwise ±1
        # OpenCV doesn't expose Shift directly; use 'a'/'d'/'w'/'s' for ±10
        elif key == 81 or key == ord('a'):  # LEFT  — cam1 -1
            idx1 = max(0, idx1 - 1)
            need_refresh = True
        elif key == 83 or key == ord('d'):  # RIGHT — cam1 +1
            idx1 = min(total1 - 1, idx1 + 1)
            need_refresh = True
        elif key == 82 or key == ord('w'):  # UP    — cam2 -1
            idx2 = max(0, idx2 - 1)
            need_refresh = True
        elif key == 84 or key == ord('s'):  # DOWN  — cam2 +1
            idx2 = min(total2 - 1, idx2 + 1)
            need_refresh = True
        elif key == ord('A'):               # SHIFT+LEFT  — cam1 -10
            idx1 = max(0, idx1 - 10)
            need_refresh = True
        elif key == ord('D'):               # SHIFT+RIGHT — cam1 +10
            idx1 = min(total1 - 1, idx1 + 10)
            need_refresh = True
        elif key == ord('W'):               # SHIFT+UP    — cam2 -10
            idx2 = max(0, idx2 - 10)
            need_refresh = True
        elif key == ord('S'):               # SHIFT+DOWN  — cam2 +10
            idx2 = min(total2 - 1, idx2 + 10)
            need_refresh = True


# ── Save synced video ──────────────────────────────────────────────────────────

def save_trimmed(video_path, start_frame, out_path):
    fps, total, width, height = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    written = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    print(f"  Saved: {out_path}  ({written} frames, started at frame {start_frame}/{total})")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for path in [VIDEO1, VIDEO2]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")

    fps1, total1, _, _ = get_video_info(VIDEO1)
    fps2, total2, _, _ = get_video_info(VIDEO2)

    print("Scanning brightness signals...")
    br1 = compute_brightness(VIDEO1)
    br2 = compute_brightness(VIDEO2)

    print(f"  cam1: {total1} frames @ {fps1:.1f} fps  "
          f"brightness: {br1.min():.0f}–{br1.max():.0f}")
    print(f"  cam2: {total2} frames @ {fps2:.1f} fps  "
          f"brightness: {br2.min():.0f}–{br2.max():.0f}")

    if abs(fps1 - fps2) > 0.5:
        print(f"  WARNING: FPS mismatch ({fps1:.1f} vs {fps2:.1f})")

    print("\nDetecting flash...")
    flash1 = find_flash_frame(br1)
    flash2 = find_flash_frame(br2)

    if flash1 is None:
        print("  ERROR: flash not detected in cam1.")
        print("  Try lowering FLASH_SIGMA at the top of the script.")
        exit(1)
    if flash2 is None:
        print("  ERROR: flash not detected in cam2.")
        print("  Try lowering FLASH_SIGMA at the top of the script.")
        exit(1)

    t1_ms = (flash1 / fps1) * 1000
    t2_ms = (flash2 / fps2) * 1000
    print(f"  cam1: frame {flash1}  ({t1_ms:.1f} ms)")
    print(f"  cam2: frame {flash2}  ({t2_ms:.1f} ms)")
    print(f"  Offset: {abs(t1_ms - t2_ms):.1f} ms")

    if SHOW_PLOT:
        show_brightness_plot(br1, br2, flash1, flash2, fps1, fps2)

    print("\nOpening preview — adjust if needed, then press ENTER...")
    print("  LEFT/RIGHT : cam1 ±1 frame")
    print("  UP/DOWN    : cam2 ±1 frame")
    print("  A/D / W/S  : ±1 frame (same, for keyboards where arrows don't work)")
    print("  SHIFT+above: ±10 frames")
    print("  ENTER      : confirm and save")
    print("  ESC        : cancel")

    confirmed1, confirmed2 = interactive_preview(
        VIDEO1, VIDEO2, flash1, flash2, br1, br2
    )

    if confirmed1 is None:
        print("Cancelled.")
        exit(0)

    print(f"\nConfirmed sync points:  cam1=frame {confirmed1},  cam2=frame {confirmed2}")

    print("\nSaving synced videos...")
    save_trimmed(VIDEO1, confirmed1, OUT_CAM1)
    save_trimmed(VIDEO2, confirmed2, OUT_CAM2)

    print("\nDone.")
    print(f"  {OUT_CAM1}")
    print(f"  {OUT_CAM2}")
