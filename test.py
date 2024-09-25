import os
import glob
import numpy as np
import cv2
from skimage import io
import tifffile
from skimage.filters import threshold_otsu

# 1. Загрузка и удаление фона
def load_and_preprocess_image(tiff_file, slice_idx):
    # Загружаем определённый срез из многослойного TIFF файла
    with tifffile.TiffFile(tiff_file) as tif:
        img = tif.pages[slice_idx].asarray()

    # Преобразуем изображение в серый цвет, если нужно
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Используем Otsu threshold для удаления фона
    thresh_val = threshold_otsu(gray)
    binary_mask = gray > thresh_val
    
    # Применяем маску, чтобы оставить только образец
    img[~binary_mask] = 0
    return img

# 2. Полуколичественный анализ
def analyze_image(img):
    lipid_mask = (img[:,:,0] > 200)  # Порог для яркости на канале R
    lipid_area = np.sum(lipid_mask)

    calcium_mask = (img[:,:,2] < 50)  # Порог для темных областей на канале B
    calcium_area = np.sum(calcium_mask)

    return lipid_area, calcium_area

# 3. Трёхмерная реконструкция
def reconstruct_3d(images, slice_thickness):
    z_stacks = np.array([img for img in images])
    return z_stacks

# Функция для обработки одного файла TIFF
def process_tiff(tiff_file):
    with tifffile.TiffFile(tiff_file) as tif:
        num_slices = len(tif.pages)  # Количество срезов

    slice_thickness = 1.0  # задаём толщину между срезами
    processed_images = []

    for i in range(num_slices):
        img = load_and_preprocess_image(tiff_file, i)
        lipid_area, calcium_area = analyze_image(img)
        processed_images.append(img)
        print(f'Slice {i}: Lipid area = {lipid_area}, Calcium area = {calcium_area}')
    
    model_3d = reconstruct_3d(processed_images, slice_thickness)
    return model_3d

# 4. Обработка всех файлов в папке
def process_directory(directory_path):
    # Находим все .tiff файлы в директории
    tiff_files = glob.glob(os.path.join(directory_path, "*.tiff"))

    # Обрабатываем каждый файл
    for tiff_file in tiff_files:
        print(f"Processing file: {tiff_file}")
        model_3d = process_tiff(tiff_file)
        # Здесь можно сохранить результат или визуализировать модель
        print(f"Finished processing {tiff_file}")

# Пример вызова:
directory_path = "images/"
process_directory(directory_path)
