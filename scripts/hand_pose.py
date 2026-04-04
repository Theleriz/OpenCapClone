import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

# --- PATH SETTINGS ---
base_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_path, '..', 'models', 'hand_landmarker.task')
INPUT_VIDEO = os.path.join(base_path, '..', 'media', 'cam2.mp4')
OUTPUT_DIR = os.path.join(base_path, '..', 'output_data')
OUTPUT_FILENAME = 'hands_coords_cam2.csv'
OUTPUT_VIDEO = 'hands_tracked_cam2.mp4'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- MEDIAPIPE TASKS INITIALIZATION ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# Связи скелета руки — без легаси
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

def get_csv_header():
    header = ['frame_idx', 'timestamp_ms']
    for h in [0, 1]:
        for p in range(21):
            header.extend([f'h{h}_p{p}_x', f'h{h}_p{p}_y'])
    return header

def draw_hand(frame, landmarks, color):
    pts = [
        (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
        for lm in landmarks
    ]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], color, 2)

    for p_idx, (cx, cy) in enumerate(pts):
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)
        cv2.putText(
            frame, str(p_idx),
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1, cv2.LINE_AA
        )

def process_video():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing started: {INPUT_VIDEO} ({total_frames} frames, {width}x{height} @ {fps:.1f}fps)")

    output_csv_path   = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    output_video_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO)

    # --- VIDEO WRITER ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    HAND_COLORS = [
        (0, 255, 255),    # рука 0 — зелёная
        (255, 100, 0),  # рука 1 — синяя
    ]
    HAND_LABELS = ['Hand 0', 'Hand 1']

    with open(output_csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(get_csv_header())

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            row_data = [0.0] * 84

            if result.hand_landmarks:
                for h_idx, landmarks in enumerate(result.hand_landmarks):
                    if h_idx > 1:
                        break

                    color = HAND_COLORS[h_idx]

                    for p_idx, lm in enumerate(landmarks):
                        pixel_x = lm.x * frame.shape[1]
                        pixel_y = lm.y * frame.shape[0]
                        row_data[h_idx * 42 + p_idx * 2] = pixel_x
                        row_data[h_idx * 42 + p_idx * 2 + 1] = pixel_y

                    draw_hand(frame, landmarks, color)

                    wrist = landmarks[0]
                    wx = int(wrist.x * frame.shape[1])
                    wy = int(wrist.y * frame.shape[0])
                    cv2.putText(
                        frame, HAND_LABELS[h_idx],
                        (wx - 20, wy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA
                    )

            writer_csv.writerow([frame_idx, timestamp_ms] + row_data)

            # Инфо на кадре
            cv2.putText(
                frame, f"Frame: {frame_idx}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 255), 2, cv2.LINE_AA
            )

            # Записываем кадр в выходное видео
            writer_video.write(frame)

            cv2.imshow('MediaPipe Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Progress: {frame_idx}/{total_frames}")

    cap.release()
    writer_video.release()
    cv2.destroyAllWindows()
    print(f"CSV saved:   {output_csv_path}")
    print(f"Video saved: {output_video_path}")

if __name__ == "__main__":
    process_video()