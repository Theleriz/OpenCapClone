import cv2
import numpy as np
import os
import json

# --- Настройки ---
# Количество ВНУТРЕННИХ углов шахматной доски (например, 9 по горизонтали, 6 по вертикали)
CHECKERBOARD = (9, 6)
# Размер одного квадрата в миллиметрах (важно для реальных масштабов 3D)
SQUARE_SIZE = 25 

# Папка с кадрами для калибровки (создай её и положи туда скриншоты с доской)
CALIB_IMAGES_PATH = '../calibration_images/cam1/'

def calibrate_camera(images_dir):
    # Критерии для уточнения координат углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Массивы для хранения точек
    objpoints = [] # 3D точки в реальном мире (0,0,0), (1,0,0)...
    imgpoints = [] # 2D точки на изображении

    # Подготовка 3D координат (0,0,0), (25,0,0), (50,0,0)...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ищем углы шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            # Уточняем координаты углов
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Рисуем для проверки
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Самый важный момент: вычисляем параметры камеры
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return mtx, dist


if __name__ == "__main__":
    # Вызываем калибровку
    mtx, dist = calibrate_camera("cam2_imgs")

    if mtx is not None:
        # 1. Подготовка данных (превращаем массивы numpy в обычные списки Python)
        data_to_save = {
            "camera_matrix": mtx.tolist(),
            "dist_coefficients": dist.tolist(),
            "status": "calibrated",
            "checkerboard": CHECKERBOARD
        }

        # 2. Создаем папку, если её нет
        output_dir = "output_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 3. Сохраняем в JSON с отступами для красоты (indent=4)
        file_path = os.path.join(output_dir, "camera_params2.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"--- Данные успешно сохранены в {file_path} ---")