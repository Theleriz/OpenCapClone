# План: Двухэтапная авто-калибровка без шахматной доски

## Context
Проект — стерео-система трекинга рук. Сейчас калибровка требует шахматную доску и видео.
Цель — полная авто-калибровка: система сама определяет K, R, T из произвольной сцены.
- **Этап 1** независим и полностью работоспособен сам по себе.
- **Этап 2** опционально дополняет Этап 1, уточняя R, T в реальном времени.

Оба этапа сохраняют те же JSON-файлы, что использует остальной пайплайн:
`camera_params1.json`, `camera_params2.json`, `matrix_vector.json`

---

## Этап 1: Feature-based Self-Calibration (SIFT + F matrix)

### Новый файл: `scripts/feature_calibrate.py`

### Алгоритм (шаги)
1. Открыть обе камеры, захватить `N` кадров из живой сцены (по умолчанию 60)
2. На каждой паре кадров: детектировать SIFT, сопоставить через FLANN, фильтровать по тесту Лоу (ratio < 0.75)
3. Аккумулировать `pts1`, `pts2` по всем кадрам
4. Оценить фундаментальную матрицу **F** через RANSAC (8-точечный алгоритм)
5. Аппроксимировать матрицу камеры **K** из размера изображения
6. Вычислить эссенциальную матрицу **E** из F и K
7. Разложить E через SVD → получить R, T (проверка хиральности)
8. Сохранить все параметры в JSON

### Формулы — Этап 1

**Эпиполярное ограничение (основа всего метода):**
```
x2ᵀ · F · x1 = 0
```
где `x1`, `x2` — однородные координаты соответствующих точек в камере 1 и 2.

**8-точечный алгоритм (оценка F):**
Для каждой пары точек строим строку матрицы A:
```
Aᵢ = [x2ₓ·x1ₓ,  x2ₓ·x1ᵧ,  x2ₓ,
       x2ᵧ·x1ₓ,  x2ᵧ·x1ᵧ,  x2ᵧ,
       x1ₓ,       x1ᵧ,       1]
```
Решение: `SVD(A)` → последний правый сингулярный вектор → reshape в 3×3 → применить constraint rank(F)=2

**Аппроксимация K из размера изображения (W×H):**
```
K = | f   0   W/2 |
    | 0   f   H/2 |     f = max(W, H)
    | 0   0   1   |
```

**Эссенциальная матрица:**
```
E = K2ᵀ · F · K1
```

**Разложение E (SVD):**
```
E = U · Σ · Vᵀ,    Σ = diag(1, 1, 0)

W = | 0  -1  0 |
    | 1   0  0 |
    | 0   0  1 |

R = U · W · Vᵀ   (или  U · Wᵀ · Vᵀ)
T = U[:, 2]       (третий столбец U)
```
Получаем 4 кандидата (R₁/R₂ × ±T). Выбираем тот, где 3D-точки оказываются перед обеими камерами (проверка хиральности). OpenCV делает это в `cv2.recoverPose()`.

**Ошибка Сэмпсона (для фильтрации выбросов при RANSAC):**
```
d_sampson = (x2ᵀ·F·x1)² / ((F·x1)₀² + (F·x1)₁² + (Fᵀ·x2)₀² + (Fᵀ·x2)₁²)
```

### OpenCV-функции Этапа 1
| Функция | Назначение |
|---------|-----------|
| `cv2.SIFT_create()` | Детектор ключевых точек |
| `cv2.FlannBasedMatcher()` | Быстрое сопоставление |
| `cv2.findFundamentalMat(..., RANSAC)` | Оценка F с отсевом выбросов |
| `cv2.findEssentialMat()` | E = K2ᵀ F K1 |
| `cv2.recoverPose()` | SVD-разложение E → R, T + cheirality |

### Выходные файлы (совместимы с остальным пайплайном)
```
output_data/camera_params1.json  ← mtx1, dist (zeros если без дисторсии)
output_data/camera_params2.json  ← mtx2, dist
output_data/matrix_vector.json   ← R, T
```

### Использование
```bash
python scripts/feature_calibrate.py
python scripts/feature_calibrate.py --cam1 0 --cam2 2 --frames 80
```

---

## Этап 2: Online Refinement (MediaPipe landmarks)

### Новый файл: `scripts/online_calibrate.py`

Загружает начальные параметры из Этапа 1 (или из chessboard-калибровки).
В реальном времени уточняет **только R, T** (экзринсики), не трогая K.

### Алгоритм (шаги)
1. Загрузить K, dist, R, T из `output_data/*.json`
2. Открыть обе камеры, запустить MediaPipe HandLandmarker на каждом кадре
3. Для каждой пары кадров: извлечь 21 точку руки из обеих камер
4. Вычислить ошибку Сэмпсона для каждой точки — отфильтровать выбросы
5. Аккумулировать "хорошие" соответствия в скользящий буфер
6. Каждые `UPDATE_EVERY` кадров: переоценить F из буфера → обновить R, T
7. Плавно обновить параметры через EMA, сохранить в JSON

### Формулы — Этап 2

**Epipolar error на каждый landmark (проверка качества соответствия):**
```
e_i = |x2ᵀ · F · x1| / √((F·x1)₀² + (F·x1)₁²)
```
Если `e_i < threshold` → точка принимается в буфер.

**Переоценка F из буфера соответствий:**
```
F_new = findFundamentalMat(pts1_buffer, pts2_buffer, RANSAC)
E_new = K2ᵀ · F_new · K1
R_new, T_new = recoverPose(E_new, pts1_buffer, pts2_buffer, K1)
```

**EMA-сглаживание обновления (через ось-угол, не через матрицу):**
```
rvec_old = Rodrigues(R_old)
rvec_new = Rodrigues(R_new)
rvec_smooth = α · rvec_new + (1-α) · rvec_old    (α = 0.3)
R_smooth = Rodrigues(rvec_smooth)

T_smooth = α · T_new + (1-α) · T_old
```
Смешивание в пространстве оси-угла (Rodrigues) корректно, т.к. пространство матриц вращения SO(3) нелинейно.

**Ошибка Сэмпсона (та же что в Этапе 1, но теперь как метрика мониторинга):**
```
d_sampson = (x2ᵀ·F·x1)² / ((F·x1)₀² + (F·x1)₁² + (Fᵀ·x2)₀² + (Fᵀ·x2)₁²)
```

### OpenCV/MediaPipe функции Этапа 2
| Функция | Назначение |
|---------|-----------|
| `vision.HandLandmarker` | 21 landmark обеих рук |
| `cv2.findFundamentalMat(..., RANSAC)` | Переоценка F из буфера |
| `cv2.findEssentialMat()` | E из F и K |
| `cv2.recoverPose()` | R, T из E |
| `cv2.Rodrigues()` | Матрица ↔ ось-угол для EMA |

### Использование
```bash
# После Этапа 1 (или любой другой калибровки):
python scripts/online_calibrate.py
python scripts/online_calibrate.py --cam1 0 --cam2 2 --alpha 0.3 --buffer 500
```

---

## Критические файлы

| Файл | Роль |
|------|------|
| `scripts/feature_calibrate.py` | **СОЗДАТЬ** — Этап 1 |
| `scripts/online_calibrate.py` | **СОЗДАТЬ** — Этап 2 |
| `scripts/auto_calibrate.py` | Существующий (chessboard-live), не трогать |
| `output_data/camera_params1.json` | Читается/перезаписывается обоими этапами |
| `output_data/camera_params2.json` | Читается/перезаписывается обоими этапами |
| `output_data/matrix_vector.json` | Читается/перезаписывается обоими этапами |
| `scripts/triangulate_3d.py` | Не трогать — потребляет JSON-файлы |
| `scripts/hand_pose.py` | Не трогать — Этап 2 копирует логику MediaPipe |

---

## Проверка (Verification)

### Этап 1
```bash
python scripts/feature_calibrate.py --cam1 0 --cam2 1
# Ожидаем: reprojection error < 2.0 px (без дисторсии будет выше, чем с шахматкой)
# Проверить: output_data/camera_params1.json, camera_params2.json, matrix_vector.json созданы
# Затем запустить triangulate_3d.py — должен работать без изменений
```

### Этап 2
```bash
python scripts/online_calibrate.py --cam1 0 --cam2 1
# Ожидаем: средняя epipolar error снижается со временем
# Проверить: matrix_vector.json обновляется каждые UPDATE_EVERY кадров
# Затем запустить triangulate_3d.py с новыми params — 3D должен быть точнее
```
