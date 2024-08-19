import os
import numpy as np
from osgeo import gdal
import random

# 全局变量，用于连续命名输出文件
global_index = 0

def has_nan_values(image_path):
    ds = gdal.Open(image_path)
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        data = band.ReadAsArray()
        if np.isnan(data).any():
            return True
    return False

def crop_and_filter_image(image_path, label_path, size=384, overlap=0.2, min_pixel_ratio=0.2, max_pixel_ratio=0.95):
    img_ds = gdal.Open(image_path)
    label_ds = gdal.Open(label_path)

    width, height = img_ds.RasterXSize, img_ds.RasterYSize
    channels = img_ds.RasterCount

    assert width == label_ds.RasterXSize and height == label_ds.RasterYSize, f"Image {image_path} and label {label_path} do not have the same dimensions."

    zero_pixel_chunks = []
    non_zero_pixel_chunks = []

    step = int(size * (1 - overlap))  # 计算步长，设置20%的重叠率

    for i in range(0, width, step):
        for j in range(0, height, step):
            img_chunk = np.zeros((channels, size, size), dtype=np.uint8)
            label_chunk = np.zeros((size, size), dtype=np.uint8)

            i_end = min(i + size, width)
            j_end = min(j + size, height)

            img_data = img_ds.ReadAsArray(i, j, i_end - i, j_end - j)
            label_data = label_ds.GetRasterBand(1).ReadAsArray(i, j, i_end - i, j_end - j)

            img_chunk[:, :img_data.shape[1], :img_data.shape[2]] = img_data
            label_chunk[:label_data.shape[0], :label_data.shape[1]] = label_data

            zero_pixel_ratio = np.sum(img_chunk == 0) / (size * size * channels)
            pixel_ratio = np.sum(label_chunk == 255) / (size * size)

            if min_pixel_ratio <= pixel_ratio <= max_pixel_ratio:
                non_zero_pixel_chunks.append((img_chunk, label_chunk))
            elif zero_pixel_ratio > 0.5:
                continue
            elif np.sum(label_chunk == 0) == size * size:
                zero_pixel_chunks.append((img_chunk, label_chunk))

    return non_zero_pixel_chunks, zero_pixel_chunks

def write_chunk_to_file(img_chunk, label_chunk, img_output_folder, label_output_folder):
    global global_index
    global_index += 1
    img_output_path = os.path.join(img_output_folder, f"{global_index}.tif")
    label_output_path = os.path.join(label_output_folder, f"{global_index}.tif")

    driver = gdal.GetDriverByName('GTiff')
    img_output_ds = driver.Create(img_output_path, 384, 384, img_chunk.shape[0], gdal.GDT_Byte)
    label_output_ds = driver.Create(label_output_path, 384, 384, 1, gdal.GDT_Byte)

    img_output_ds.SetGeoTransform([0, 1, 0, 0, 0, -1])  # Example geotransform, adjust as needed
    img_output_ds.SetProjection("EPSG:4326")  # Example projection, adjust as needed
    label_output_ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    label_output_ds.SetProjection("EPSG:4326")

    for band in range(img_chunk.shape[0]):
        img_output_ds.GetRasterBand(band + 1).WriteArray(img_chunk[band])

    label_output_ds.GetRasterBand(1).WriteArray(label_chunk)

    img_output_ds = None
    label_output_ds = None

def main(img_input_folder, label_input_folder, img_output_folder, label_output_folder):
    global global_index
    img_files = sorted([f for f in os.listdir(img_input_folder) if f.endswith('.tif')])
    label_files = sorted([f for f in os.listdir(label_input_folder) if f.endswith('.tif')])

    assert len(img_files) == len(label_files), "Mismatch in number of image and label files."

    pairs = list(zip(img_files, label_files))
    random.shuffle(pairs)

    zero_count = 0
    non_zero_count = 0

    for img_file, label_file in pairs:
        img_path = os.path.join(img_input_folder, img_file)
        label_path = os.path.join(label_input_folder, label_file)

        assert not has_nan_values(img_path), f"Image {img_path} contains NaN values."
        assert not has_nan_values(label_path), f"Label {label_path} contains NaN values."

        non_zero_chunks, zero_chunks = crop_and_filter_image(img_path, label_path)
        zero_count += len(zero_chunks)
        non_zero_count += len(non_zero_chunks)

        combined_chunks = non_zero_chunks + zero_chunks

        # 写入文件
        for img_chunk, label_chunk in combined_chunks:
            write_chunk_to_file(img_chunk, label_chunk, img_output_folder, label_output_folder)

    total_samples = zero_count + non_zero_count
    if total_samples > 0:
        zero_percentage = (zero_count / total_samples) * 100
        non_zero_percentage = (non_zero_count / total_samples) * 100
    else:
        zero_percentage = non_zero_percentage = 0

    print(f"全零像素样本总数: {zero_count} ({zero_percentage:.2f}%)")
    print(f"非零像素样本总数: {non_zero_count} ({non_zero_percentage:.2f}%)")

if __name__ == "__main__":
    img_input_folder = r'E:\data\trans_gd\original\img'
    label_input_folder = r'E:\data\trans_gd\original\lbl'
    img_output_folder = r'E:\data\trans_gd\image'
    label_output_folder = r'E:\data\trans_gd\label'

    if not os.path.exists(img_output_folder):
        os.makedirs(img_output_folder)
    if not os.path.exists(label_output_folder):
        os.makedirs(label_output_folder)

    main(img_input_folder, label_input_folder, img_output_folder, label_output_folder)
