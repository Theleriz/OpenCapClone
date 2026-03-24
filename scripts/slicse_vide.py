import cv2
import os

def save_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Сохраняем каждый 15-й кадр (примерно каждые 0.5 сек при 30 FPS)
        if frame_id % 15 == 0:
            cv2.imwrite(f"{output_folder}/calib_{count}.jpg", frame)
            count += 1
        frame_id += 1
    
    cap.release()
    print(f"Сохранено {count} кадров в {output_folder}")


if __name__ == "__main__":
    save_frames("../media/calibration_cam1", "../media")
# Запустите для обоих видео
# save_frames('../videos/calib_cam1.mp4', '../calibration_images/cam1')
# save_frames('../videos/calib_cam2.mp4', '../calibration_images/cam2')