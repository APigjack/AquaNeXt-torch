import os
import numpy as np
from osgeo import gdal
import cv2
import random
from PIL import Image, ImageEnhance, ImageFilter

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(f"{fileName} 文件无法打开")
        return None
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    data = dataset.ReadAsArray(0, 0, width, height)
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

def writeTiff(im_data, im_geotrans, im_proj, path):
    datatype = gdal.GDT_Byte
    bands = im_data.shape[0] if len(im_data.shape) == 3 else 1
    if bands == 1:
        im_data = np.array([im_data])
    height, width = im_data.shape[1:]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, width, height, bands, datatype)
    if dataset:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset
    else:
        print("无法创建文件：", path)
        return False
    return True

def shuffle_files(image_path, label_path):
    images = os.listdir(image_path)
    labels = os.listdir(label_path)

    combined = list(zip(images, labels))
    random.shuffle(combined)

    new_image_names = [f"{idx+1}{os.path.splitext(img)[1]}" for idx, (img, _) in enumerate(combined)]
    new_label_names = [f"{idx+1}{os.path.splitext(lbl)[1]}" for idx, (_, lbl) in enumerate(combined)]

    temp_image_path = os.path.join(image_path, "temp")
    temp_label_path = os.path.join(label_path, "temp")

    os.makedirs(temp_image_path, exist_ok=True)
    os.makedirs(temp_label_path, exist_ok=True)

    for img, new_name in zip(images, new_image_names):
        os.rename(os.path.join(image_path, img), os.path.join(temp_image_path, new_name))

    for lbl, new_name in zip(labels, new_label_names):
        os.rename(os.path.join(label_path, lbl), os.path.join(temp_label_path, new_name))

    for img, new_name in zip(os.listdir(temp_image_path), new_image_names):
        os.rename(os.path.join(temp_image_path, img), os.path.join(image_path, new_name))

    for lbl, new_name in zip(os.listdir(temp_label_path), new_label_names):
        os.rename(os.path.join(temp_label_path, lbl), os.path.join(label_path, new_name))

    os.rmdir(temp_image_path)
    os.rmdir(temp_label_path)

def apply_color_jitter(image):
    if image.shape[0] < 3:
        raise ValueError("Image must have at least 3 channels for RGB conversion")
    img = Image.fromarray(np.stack((image[0], image[1], image[2]), axis=-1))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.3, 1.7))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    return np.stack([np.array(img)[..., i] for i in range(3)])

def apply_gaussian_blur(image):
    if image.shape[0] < 3:
        raise ValueError("Image must have at least 3 channels for RGB conversion")
    img = Image.fromarray(np.stack((image[0], image[1], image[2]), axis=-1))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 2)))
    return np.stack([np.array(img)[..., i] for i in range(3)])

train_image_path = r'E:\data\trans_gd\image'
train_label_path = r'E:\data\trans_gd\label'

imageList = os.listdir(train_image_path)
labelList = os.listdir(train_label_path)
tran_num = len(imageList)  # Start tran_num based on existing images

for i in range(len(imageList)):
    img_file = os.path.join(train_image_path, imageList[i])
    label_file = os.path.join(train_label_path, labelList[i])

    read_result = readTif(img_file)
    if read_result is None:
        continue
    _, _, _, im_data, im_geotrans, im_proj = read_result
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    if label is None:
        continue
    label = label.astype(np.uint8)

    if np.any(label == 255):
        augmentations = [
            (apply_color_jitter(im_data), label),  # Color jitter
            (np.flip(im_data, axis=1), cv2.flip(label, 0)),  # Vertical flip
            (np.flip(np.flip(im_data, axis=1), axis=2), cv2.flip(label, -1)),  # Diagonal flip
            (apply_gaussian_blur(im_data), label),  # Gaussian blur
            (np.flip(im_data, axis=2), cv2.flip(label, 1))  # Horizontal flip
        ]

        for aug_data, aug_label in augmentations:
            tran_num += 1
            img_path = os.path.join(train_image_path, f'{tran_num}{imageList[i][-4:]}')
            label_path = os.path.join(train_label_path, f'{tran_num}{labelList[i][-4:]}')
            if writeTiff(aug_data, im_geotrans, im_proj, img_path):
                cv2.imwrite(label_path, aug_label)

# After all processing is done, shuffle the files
shuffle_files(train_image_path, train_label_path)
