import cv2
import numpy as np
import os
import json

# --- Твои данные из предыдущего шага (вставь свои цифры здесь!) ---
mtx1 = np.array([[
            1097.5840828812597,
            0.0,
            1025.247734218068
        ],
        [
            0.0,
            984.0683528693361,
            385.81068894476743
        ],
        [
            0.0,
            0.0,
            1.0
        ]]) # Пример
dist1 = np.array([
            -0.48659317148416364,
            0.8148522049914797,
            -0.012198322959003607,
            0.041759706725023424,
            -0.5303064734026546
        ])

mtx2 = np.array([[
            1381.2870410610528,
            0.0,
            609.2616368532443
        ],
        [
            0.0,
            1410.1934137649462,
            380.89647173154975
        ],
        [
            0.0,
            0.0,
            1.0
        ]]) # Пример
dist2 = np.array([
            0.3277110794081981,
            -10.040639694630121,
            -0.008637784469658266,
            -0.08391471654141584,
            49.07735821474451
        ])

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 25 # мм

def stereo_calibrate(dir1, dir2):
    objpoints = [] 
    imgpoints1 = [] 
    imgpoints2 = [] 

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    images1 = sorted([os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.jpg')])
    images2 = sorted([os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith('.jpg')])

    for img1_path, img2_path in zip(images1, images2):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)

        if ret1 and ret2:
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

    # Вычисляем взаимное расположение камер
    flags = cv2.CALIB_FIX_INTRINSIC # Не меняем внутренние параметры, они уже есть
    ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, 
        gray1.shape[::-1], criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5), 
        flags=flags)

    return R, T

R, T = stereo_calibrate('cam1_imgs', 'cam2_imgs')
if R is not None:
        # 1. Подготовка данных (превращаем массивы numpy в обычные списки Python)
        data_to_save = {
            "rotation_matrix": R.tolist(),
            "vector": T.tolist(),
        }

        # 2. Создаем папку, если её нет
        output_dir = "output_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 3. Сохраняем в JSON с отступами для красоты (indent=4)
        file_path = os.path.join(output_dir, "matrix_vector.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)