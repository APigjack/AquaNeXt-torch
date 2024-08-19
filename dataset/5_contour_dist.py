import numpy as np
import os
import cv2
from osgeo import gdal

def read_img(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data

def write_img(filename, im_proj, im_geotrans, im_data):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    datatype = gdal.GDT_Byte if 'int8' in im_data.dtype.name else gdal.GDT_UInt16 if 'int16' in im_data.dtype.name else gdal.GDT_Float32
    im_bands = 1 if len(im_data.shape) == 2 else im_data.shape[0]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_data.shape[-1], im_data.shape[-2], im_bands, datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data if im_bands == 1 else im_data[i])
        del dataset
    else:
        raise Exception(f"无法创建文件: {filename}")

maskRoot = r"E:\data\trans_gd\mask"
distRoot = r"E:\data\trans_gd\dist_contour_tif"
boundaryRoot = r"E:\data\trans_gd\contour"

for imgPath in os.listdir(maskRoot):
    input_path = os.path.join(maskRoot, imgPath)
    boundaryOutPath = os.path.join(boundaryRoot, imgPath)
    distOutPath = os.path.join(distRoot, imgPath)
    im_proj, im_geotrans, im_width, im_height, im_data = read_img(input_path)

    # 欧几里得
    result = cv2.distanceTransform(src=im_data, distanceType=cv2.DIST_L2, maskSize=3)

    # 使用曼哈顿距离（L1 距离）进行距离变换
    # result = cv2.distanceTransform(src=im_data, distanceType=cv2.DIST_L1, maskSize=3)

    min_value = np.min(result)
    max_value = np.max(result)
    if max_value > min_value:
        scaled_image = ((result - min_value) / (max_value - min_value)) * 255
        result = scaled_image.astype(np.uint8)
    else:
        result = np.zeros_like(im_data, dtype=np.uint8)  # 如果没有有效的距离变换结果，创建一个全零图像
    write_img(distOutPath, im_proj, im_geotrans, result)
    boundary = cv2.Canny(im_data, 100, 200)
    write_img(boundaryOutPath, im_proj, im_geotrans, boundary)
