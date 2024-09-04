import os
import shutil
import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import DataLoader
from dataset import DatasetImageOnly
from tqdm import tqdm
import cv2
import glob
import rasterio

from AquaNeXt import AquaNeXt



def create_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_tile_indices(width, height, tile_size, overlap):
    step_size = int(tile_size * (1 - overlap))
    indices = []
    for j in range(0, height, step_size):
        for i in range(0, width, step_size):
            indices.append((i, j))
    return indices


def save_tile(tile, output_path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, tile.shape[1], tile.shape[0], tile.shape[2], gdal.GDT_Byte)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    for i in range(tile.shape[2]):
        out_ds.GetRasterBand(i + 1).WriteArray(tile[:, :, i])
    out_ds.FlushCache()
    out_ds = None


def build_model(model_type):
    if model_type == "AquaNeXt":
        model = AquaNeXt(num_classes=2, use_ms_cam=False)
        is_multi_task = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, is_multi_task


def apply_mask(image, mask, overlay=False, color=(0, 105, 148), alpha=1):
    if overlay:
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
    else:
        image = (mask * 255).astype(np.uint8)
    return image


def save_georeferenced_image(image, reference_path, save_path):
    reference_ds = gdal.Open(reference_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(save_path, image.shape[1], image.shape[0], image.shape[2], gdal.GDT_Byte)
    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())
    for i in range(image.shape[2]):
        out_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])
    out_ds.FlushCache()
    out_ds = None
    reference_ds = None


def get_raster_info(raster_path):
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
    return transform, crs, width, height


def restore_labels(original_raster_path, predicted_labels_path, output_path, label_type):
    transform, crs, width, height = get_raster_info(original_raster_path)
    restored_labels = np.ones((height, width), dtype=np.uint8) * 255 if label_type != 'dist' else np.zeros(
        (height, width), dtype=np.float32)
    label_files = glob.glob(os.path.join(predicted_labels_path, '*.tif'))
    for label_file in tqdm(label_files, desc=f"Restoring {label_type} labels"):
        with rasterio.open(label_file) as src:
            labels = src.read(1)
            label_transform = src.transform
            label_height, label_width = labels.shape
            x_offset = int((label_transform.c - transform.c) / transform.a)
            y_offset = int((label_transform.f - transform.f) / transform.e)
            x_end = min(x_offset + label_width, width)
            y_end = min(y_offset + label_height, height)
            labels_cropped = labels[:y_end - y_offset, :x_end - x_offset]
            if label_type == 'dist':
                restored_labels[y_offset:y_end, x_offset:x_end] = np.maximum(
                    restored_labels[y_offset:y_end, x_offset:x_end], labels_cropped)
            else:
                restored_labels[y_offset:y_end, x_offset:x_end] = labels_cropped
    output_file = os.path.join(output_path, f'{label_type}.tif')
    with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=restored_labels.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(restored_labels, 1)


def predict_and_save(model, inputs, device, save_path, task, is_multi_task):
    with torch.no_grad():
        outputs = model(inputs.to(device))
    if is_multi_task:
        if task == 'mask':
            result = np.argmax(outputs[0].cpu().numpy(), axis=1).squeeze()
        elif task == 'edge':
            result = np.argmax(outputs[1].cpu().numpy(), axis=1).squeeze()
        elif task == 'dist':
            result = outputs[2].cpu().numpy().squeeze()
    else:
        result = np.argmax(outputs.cpu().numpy(), axis=1).squeeze()
    np.save(save_path, result)


def load_result(result_path):
    return np.load(result_path)


def main():
    # 设置参数
    image_path = r'D:\Paper_code_pt\new\JM_regular\img.tif'
    label_path = r'D:\Paper_code_pt\new\JM_regular\lbl.tif'
    # model_file = r'D:\Paper_code_pt\new\JM_regular\AquaNeXt\AquaNeXt_NM_gd_regular\best_val_model_epoch_150.pt'
    model_file = r'D:\Paper_code_pt\new\JM_regular\AquaNeXt\AquaNeXt_NM_gd_regular\best_val_model_epoch_150.pt'
    output_folder = r'D:\Paper_code_pt\new\JM_regular\AquaNeXt\out'
    tile_size = 256
    overlap = 0.2
    model_type = 'AquaNeXt'
    predict_edge = False
    predict_dist = False
    reconstruct = True

    temp_folder = "temp"
    temp_result_folder1 = os.path.join(output_folder, 'temp_results1')
    temp_result_folder2 = os.path.join(output_folder, 'temp_results2')
    temp_result_folder3 = os.path.join(output_folder, 'temp_results3')
    create_output_directory(temp_folder)
    create_output_directory(temp_result_folder1)
    create_output_directory(temp_result_folder2)
    create_output_directory(temp_result_folder3)
    create_output_directory(output_folder)
    output_mask_folder = os.path.join(output_folder, "mask")
    output_edge_folder = os.path.join(output_folder, "edge") if predict_edge else None
    output_dist_folder = os.path.join(output_folder, "dist") if predict_dist else None
    create_output_directory(output_mask_folder)
    if predict_edge:
        create_output_directory(output_edge_folder)
    if predict_dist:
        create_output_directory(output_dist_folder)

    image_ds = gdal.Open(image_path)
    image = image_ds.ReadAsArray()
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (1, 2, 0))
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()

    label_ds = None
    label = None
    if label_path:
        label_ds = gdal.Open(label_path)
        label = label_ds.ReadAsArray()
        if label.ndim == 2:
            label = np.expand_dims(label, axis=0)
        label = np.transpose(label, (1, 2, 0))

    indices = get_tile_indices(image.shape[1], image.shape[0], tile_size, overlap)
    tile_count = 1
    for i, j in indices:
        image_tile = image[j:j + tile_size, i:i + tile_size]
        if image_tile.shape[0] != tile_size or image_tile.shape[1] != tile_size:
            pad_height = tile_size - image_tile.shape[0]
            pad_width = tile_size - image_tile.shape[1]
            image_tile = np.pad(image_tile, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

        tile_geo_transform = (
            geo_transform[0] + i * geo_transform[1],
            geo_transform[1],
            geo_transform[2],
            geo_transform[3] + j * geo_transform[5],
            geo_transform[4],
            geo_transform[5]
        )
        image_tile_path = os.path.join(temp_folder, f'image_{tile_count}.tif')
        save_tile(image_tile, image_tile_path, tile_geo_transform, projection)
        tile_count += 1

    img_names = [f[:-4] for f in os.listdir(temp_folder) if f.startswith('image') and f.endswith('.tif')]
    valLoader = DataLoader(DatasetImageOnly(temp_folder, img_names), batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, is_multi_task = build_model(model_type)
    model = model.to(device)

    state_dict = torch.load(model_file, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 module. 前缀
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    for img_file_name, inputs in tqdm(valLoader, desc="First prediction"):
        predict_and_save(model, inputs, device, os.path.join(temp_result_folder1, img_file_name[0] + '_mask.npy'),
                         'mask', is_multi_task)
        if predict_edge:
            predict_and_save(model, inputs, device, os.path.join(temp_result_folder1, img_file_name[0] + '_edge.npy'),
                             'edge', is_multi_task)
        if predict_dist:
            predict_and_save(model, inputs, device, os.path.join(temp_result_folder1, img_file_name[0] + '_dist.npy'),
                             'dist', is_multi_task)

    for img_file_name, inputs in tqdm(valLoader, desc="Second prediction"):
        predict_and_save(model, inputs, device, os.path.join(temp_result_folder2, img_file_name[0] + '_mask.npy'),
                         'mask', is_multi_task)
        if predict_edge:
            predict_and_save(model, inputs, device, os.path.join(temp_result_folder2, img_file_name[0] + '_edge.npy'),
                             'edge', is_multi_task)
        if predict_dist:
            predict_and_save(model, inputs, device, os.path.join(temp_result_folder2, img_file_name[0] + '_dist.npy'),
                             'dist', is_multi_task)

    for img_file_name, inputs in tqdm(valLoader, desc="Third prediction"):
        predict_and_save(model, inputs, device, os.path.join(temp_result_folder3, img_file_name[0] + '_mask.npy'),
                         'mask', is_multi_task)
        if predict_edge:
            predict_and_save(model, inputs, device, os.path.join(temp_result_folder3, img_file_name[0] + '_edge.npy'),
                             'edge', is_multi_task)
        if predict_dist:
            predict_and_save(model, inputs, device, os.path.join(temp_result_folder3, img_file_name[0] + '_dist.npy'),
                             'dist', is_multi_task)

    diff_img_names = []
    for img_file_name in tqdm(img_names, desc="Comparing results"):
        mask1 = load_result(os.path.join(temp_result_folder1, img_file_name + '_mask.npy'))
        mask2 = load_result(os.path.join(temp_result_folder2, img_file_name + '_mask.npy'))
        mask3 = load_result(os.path.join(temp_result_folder3, img_file_name + '_mask.npy'))

        pixel_difference_1_2 = np.sum(mask1 != mask2) / mask1.size
        pixel_difference_1_3 = np.sum(mask1 != mask3) / mask1.size
        pixel_difference_2_3 = np.sum(mask2 != mask3) / mask1.size

        if pixel_difference_1_2 > 0.005:
            diff_img_names.append(img_file_name)
            final_mask = mask1 if pixel_difference_1_3 < pixel_difference_2_3 else mask2
        else:
            final_mask = mask1

        original_image_path = os.path.join(temp_folder, img_file_name + '.tif')
        original_image = cv2.imread(original_image_path)
        non_black_indices = (original_image != 0).all(axis=-1)
        if final_mask.ndim == 2:
            final_mask = np.expand_dims(final_mask, axis=2)
        final_mask[~non_black_indices] = 0

        binary_image = apply_mask(original_image.copy(), final_mask.squeeze() == 1, overlay=False)
        save_path_mask = os.path.join(output_mask_folder, img_file_name + '.tif')
        save_georeferenced_image(np.expand_dims(binary_image, axis=2), original_image_path, save_path_mask)

        if predict_edge:
            edge1 = load_result(os.path.join(temp_result_folder1, img_file_name + '_edge.npy'))
            edge2 = load_result(os.path.join(temp_result_folder2, img_file_name + '_edge.npy'))
            edge3 = load_result(os.path.join(temp_result_folder3, img_file_name + '_edge.npy'))
            final_edge = edge1 if pixel_difference_1_3 < pixel_difference_2_3 else edge2
            final_edge[~non_black_indices] = 0
            edge_color_image = np.zeros((final_edge.shape[0], final_edge.shape[1], 3), dtype=np.uint8)
            edge_color_image[final_edge == 1] = [255, 255, 255]
            save_path_edge = os.path.join(output_edge_folder, img_file_name + '.tif')
            save_georeferenced_image(edge_color_image, original_image_path, save_path_edge)

        if predict_dist:
            dist1 = load_result(os.path.join(temp_result_folder1, img_file_name + '_dist.npy'))
            dist2 = load_result(os.path.join(temp_result_folder2, img_file_name + '_dist.npy'))
            dist3 = load_result(os.path.join(temp_result_folder3, img_file_name + '_dist.npy'))
            final_dist = dist1 if pixel_difference_1_3 < pixel_difference_2_3 else dist2
            final_dist[~non_black_indices] = 0
            if np.max(final_dist) != 0:
                final_dist = (final_dist * 255 / np.max(final_dist)).astype('uint8')
            else:
                final_dist = np.zeros_like(final_dist, dtype='uint8')
            distance_colored = cv2.applyColorMap(final_dist, cv2.COLORMAP_JET)
            save_path_dist = os.path.join(output_dist_folder, img_file_name + '.tif')
            save_georeferenced_image(distance_colored, original_image_path, save_path_dist)

    if diff_img_names:
        print("Images with significant pixel difference between first two predictions:")
        for name in diff_img_names:
            print(name)

    if reconstruct:
        final_results_folder = os.path.join(output_folder)
        create_output_directory(final_results_folder)
        restore_labels(image_path, output_mask_folder, final_results_folder, 'mask')
        if predict_edge:
            restore_labels(image_path, output_edge_folder, final_results_folder, 'edge')
        if predict_dist:
            restore_labels(image_path, output_dist_folder, final_results_folder, 'dist')

    shutil.rmtree(temp_folder)
    shutil.rmtree(temp_result_folder1)
    shutil.rmtree(temp_result_folder2)
    shutil.rmtree(temp_result_folder3)


if __name__ == '__main__':
    main()
