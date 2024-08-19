import os
from osgeo import gdal, ogr, osr
import math

def find_first_file_with_extension(directory, extension):
    """在指定目录中找到第一个指定扩展名的文件"""
    for file in os.listdir(directory):
        if file.lower().endswith(extension.lower()):
            return os.path.join(directory, file)
    return None


def align_with_shp(ds_tiff, shp_layer):
    """调整栅格数据的GeoTransform以与SHP文件对齐"""
    shp_extent = shp_layer.GetExtent()

    # 计算新的GeoTransform
    gt = list(ds_tiff.GetGeoTransform())
    gt[0] = math.floor(shp_extent[0] / gt[1]) * gt[1]  # 调整X起点
    gt[3] = math.ceil(shp_extent[3] / -gt[5]) * -gt[5]  # 调整Y起点

    return gt

def convert_shp_to_tiff(shp_path, ref_tiff_path):
    """将Shapefile转换为与参考TIFF同样格式和投影的内存中TIFF数据集，并将多边形内部像素设置为255"""
    ds_tiff = gdal.Open(ref_tiff_path)
    if not ds_tiff:
        raise RuntimeError(f"无法打开参考TIFF文件：{ref_tiff_path}")

    geo_transform = ds_tiff.GetGeoTransform()
    projection = ds_tiff.GetProjection()
    x_size = ds_tiff.RasterXSize
    y_size = ds_tiff.RasterYSize

    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp = driver.Open(shp_path)
    if not shp:
        raise RuntimeError(f"无法打开Shapefile文件：{shp_path}")
    layer = shp.GetLayer()

    # 创建一个内存数据集
    target_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()

    # 为了使栅格化与原始SHP文件对齐，我们需要考虑像元中心
    gdal.RasterizeLayer(target_ds, [1], layer, options=["-at"], burn_values=[255])

    extent = layer.GetExtent()
    return target_ds, extent


def crop_image(ds, output_path, extent, geo_transform):
    """根据给定的包围框裁剪图像"""
    x_offset = int((extent[0] - geo_transform[0]) / geo_transform[1])
    y_offset = int((extent[3] - geo_transform[3]) / geo_transform[5])
    x_size = abs(int((extent[1] - extent[0]) / geo_transform[1]))
    y_size = abs(int((extent[2] - extent[3]) / geo_transform[5]))

    if x_size == 0 or y_size == 0:
        print("裁剪区域尺寸为零，跳过。")
        return

    # 读取裁剪区域
    cropped = ds.ReadAsArray(x_offset, y_offset, x_size, y_size)
    if cropped is None:
        print("读取数据失败，可能超出边界。")
        return

    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    raster_count = ds.RasterCount
    out_ds = driver.Create(output_path, x_size, y_size, raster_count, ds.GetRasterBand(1).DataType)
    out_ds.SetGeoTransform((extent[0], geo_transform[1], 0, extent[3], 0, geo_transform[5]))
    out_ds.SetProjection(ds.GetProjection())

    if raster_count == 1:
        out_ds.GetRasterBand(1).WriteArray(cropped)
    else:
        for i in range(raster_count):
            out_ds.GetRasterBand(i + 1).WriteArray(cropped[i, :, :])

    out_ds.FlushCache()
    out_ds = None
    print(f"裁剪完成并保存到：{output_path}")


tif_folder = r"C:\Users\Administrator\Desktop\paper_results\gd_trans\384_0.15\ma"
shp_folder = r"D:\project\dataset\data\gd_trans_test"
output_folder = r"C:\Users\Administrator\Desktop\paper_results\gd_trans\384_0.15\ma"   # 这里文件夹必须存在，不会自动创建

tif_file = find_first_file_with_extension(tif_folder, '.tif')
shp_file = find_first_file_with_extension(shp_folder, '.shp')

if tif_file and shp_file:
    label_ds, extent = convert_shp_to_tiff(shp_file, tif_file)
    crop_image(gdal.Open(tif_file), os.path.join(output_folder, "img.tif"), extent, gdal.Open(tif_file).GetGeoTransform())
    crop_image(label_ds, os.path.join(output_folder, "lbl.tif"), extent, gdal.Open(tif_file).GetGeoTransform())
else:
    print("Could not find required TIFF or SHP files.")
