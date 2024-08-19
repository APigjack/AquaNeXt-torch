import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
from skimage import io
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import os
from osgeo import gdal
import random


USE_RANDOM_ERASING = True  # 是否使用随机擦除


### Reading and saving of remote sensing images (Keep coordinate information)
def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if data_width == 0 and data_height == 0:
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


# 保存遥感影像
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


class DatasetImageOnly(Dataset):
    """数据集定义，仅加载影像用于预测"""

    def __init__(self, dir, file_names):
        """
        初始化数据集
        :param dir: 存放影像的文件夹路径
        :param file_names: 影像文件名列表，不包含扩展名
        """
        self.file_names = file_names
        self.dir = dir

    def __len__(self):
        """
        返回数据集中的样本数
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本的索引
        :return: 元组，包含文件名和影像数据
        """
        img_file_name = self.file_names[idx]
        image = load_image(os.path.join(self.dir, img_file_name + '.tif'))
        return img_file_name, image


class DatasetImageMaskContourDist(Dataset):
    def __init__(self, dir, file_names, distance_type):
        self.file_names = file_names
        self.distance_type = distance_type
        self.dir = dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(os.path.join(self.dir, img_file_name + '.tif'))
        mask = load_mask(os.path.join(self.dir, img_file_name + '.tif'))
        contour = load_contour(os.path.join(self.dir, img_file_name + '.tif'))
        dist = load_distance(os.path.join(self.dir, img_file_name + '.tif'), self.distance_type)
        return img_file_name, image, mask, contour, dist


def check_for_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"NaN value found in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf value found in {name}")


def load_image(path):
    img = Image.open(path)
    data_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if USE_RANDOM_ERASING:
        data_transforms.append(
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))

    data_transforms = transforms.Compose(data_transforms)
    img = data_transforms(img)
    check_for_nan_inf(img, "image")
    return img


def load_mask(path):
    mask = cv2.imread(path.replace("image", "mask").replace("tif", "tif"), 0)
    mask[mask == 255] = 1
    mask[mask == 0] = 0
    mask = torch.from_numpy(np.expand_dims(mask, 0)).long()
    check_for_nan_inf(mask, "mask")
    return mask


def load_contour(path):
    contour = cv2.imread(path.replace("image", "contour").replace("tif", "tif"), 0)
    contour[contour == 255] = 1
    contour[contour == 0] = 0
    contour = torch.from_numpy(np.expand_dims(contour, 0)).long()
    check_for_nan_inf(contour, "contour")
    return contour


def load_distance(path, distance_type):
    if distance_type == "dist_mask":
        path = path.replace("image", "dist_mask").replace("tif", "mat")
        dist = io.loadmat(path)["D2"]
    elif distance_type == "dist_contour":
        path = path.replace("image", "dist_contour").replace("tif", "mat")
        dist = io.loadmat(path)["D2"]
    elif distance_type == "dist_contour_tif":
        dist = cv2.imread(path.replace("image", "dist_contour_tif").replace("tif", "tif"), 0)
        dist = dist / 255.
    dist = torch.from_numpy(np.expand_dims(dist, 0)).float()
    check_for_nan_inf(dist, "distance")
    return dist
