import cv2
import mediapipe as mp
import csv

# --- Настройки ---
# Введите путь к вашему видео
INPUT_VIDEO = 'path/to/your/video_cam1.mp4' 
# Введите имя файла для сохранения результатов
OUTPUT_CSV = 'coords_cam1.csv' 
# Показывать ли видео в процессе обработки (сильно замедляет)
SHOW_VIDEO = True 

# --- Инициализация MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Настройка: static_image_mode=False (для видео), высокая уверенность
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,       # 0, 1, or 2 (higher is more accurate but slower)
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Список имен ключевых точек (для заголовка CSV) ---
# MediaPipe выдает 33 точки. Мы создадим заголовок: point0_x, point0_y, point1_x...
landmark_names = [f"point{i}_{coord}" for i in range(33) for coord in ['x', 'y']]
csv_header = ['frame_number'] + landmark_names

def process_video():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {INPUT_VIDEO}")
        return

    # Получаем параметры видео для логирования
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Обработка видео: {INPUT_VIDEO}")
    print(f"Всего кадров: {frame_count}, FPS: {fps:.2f}")

    # Создаем/открываем CSV файл для записи
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header) # Записываем заголовок

        current_frame = 0
        while cap.isOpened():
            success, frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # Читаем кадр. Пропускаем, если видео закончилось
            success, frame = cap.read()
            if not success:
                break

            current_frame += 1
            if current_frame % 50 == 0:
                print(f"Обработан кадр {current_frame}/{frame_count}")

            # 1. Конвертация BGR в RGB (MediaPipe требует RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Обработка кадра MediaPipe
            results = pose.process(rgb_frame)

            # 3. Извлечение координат
            frame_row = [current_frame] # Начинаем строку данных с номера кадра

            if results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in results.pose_landmarks.landmark:
                    # MediaPipe выдает нормализованные координаты (0.0 - 1.0)
                    # Превращаем их в пиксельные координаты изображения
                    # landmark.z - это относительная глубина (ее пока не используем)
                    pixel_x = lm.x * w
                    pixel_y = lm.y * h
                    frame_row.extend([pixel_x, pixel_y])
                
                # Если SHOW_VIDEO=True, рисуем точки на кадре
                if SHOW_VIDEO:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                # Если человек не обнаружен, заполняем строку нулями (или None)
                frame_row.extend([0.0] * (33 * 2))

            # 4. Запись строки данных в CSV
            writer.writerow(frame_row)

            # Отрисовка (если включена)
            if SHOW_VIDEO:
                cv2.imshow('MediaPipe Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    print(f"Готово. Координаты сохранены в {OUTPUT_CSV}")

if __name__ == "__main__":
    process_video()