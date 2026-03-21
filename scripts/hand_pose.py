import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

# --- НАСТРОЙКИ ПУТЕЙ ---
# Скрипт подстраивается под структуру: /scripts/hand_pose.py и /models/hand_landmarker.task
base_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_path, '..', 'models', 'hand_landmarker.task')
INPUT_VIDEO = os.path.join(base_path, '..', 'media', 'Cam_1.mp4') # Поменяйте на cam2.mp4 для второго запуска
OUTPUT_DIR = os.path.join(base_path, '..', 'output_data')
OUTPUT_FILENAME = 'hands_coords_cam1.csv' # Поменяйте на hands_coords_cam2.csv

# Проверка наличия модели
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден по пути: {MODEL_PATH}")

# Создание папки для вывода
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- ИНИЦИАЛИЗАЦИЯ MEDIAPIPE TASKS ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

def get_csv_header():
    # 2 руки (h0, h1), в каждой 21 точка (p0-p20), для каждой x и y
    header = ['frame_idx', 'timestamp_ms']
    for h in [0, 1]:
        for p in range(21):
            header.extend([f'h{h}_p{p}_x', f'h{h}_p{p}_y'])
    return header

def process_video():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {INPUT_VIDEO}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Начало обработки: {INPUT_VIDEO} ({total_frames} кадров)")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(get_csv_header())

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Конвертация кадра для MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Расчет временной метки (важно для режима VIDEO)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            # Детекция
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Подготовка строки данных (84 координаты нулями по умолчанию)
            row_data = [0.0] * 84 
            
            if result.hand_landmarks:
                # Проходим по найденным рукам (макс 2)
                for h_idx, landmarks in enumerate(result.hand_landmarks):
                    if h_idx > 1: break 
                    
                    for p_idx, lm in enumerate(landmarks):
                        # Переводим нормализованные координаты в пиксели
                        pixel_x = lm.x * frame.shape[1]
                        pixel_y = lm.y * frame.shape[0]
                        # Записываем в нужную позицию в списке
                        row_data[h_idx * 42 + p_idx * 2] = pixel_x
                        row_data[h_idx * 42 + p_idx * 2 + 1] = pixel_y

                # Визуализация (опционально)
                for landmarks in result.hand_landmarks:
                    for lm in landmarks:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            writer.writerow([frame_idx, timestamp_ms] + row_data)
            
            # Показ прогресса
            cv2.imshow('MediaPipe Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Прогресс: {frame_idx}/{total_frames}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Готово! Результаты сохранены в: {output_path}")

if __name__ == "__main__":
    process_video()