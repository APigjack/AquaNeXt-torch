import os
from osgeo import gdal
from shutil import copy2

# 路径设置
src_images_dir = r'E:\data\trans_gd\image'  # 训练影像数据文件夹
src_labels_dir = r'E:\data\trans_gd\label'    # 标签文件夹
dst_images_dir = r'E:\data\trans_gd\im'  # 输出影像数据文件夹
dst_labels_dir = r'E:\data\trans_gd\lb'  # 输出标签文件夹

# 确保输出目录存在
os.makedirs(dst_images_dir, exist_ok=True)
os.makedirs(dst_labels_dir, exist_ok=True)

# 读取文件名
image_files = sorted(os.listdir(src_images_dir))
label_files = sorted(os.listdir(src_labels_dir))

# 分类全负样本和非全负样本
all_negatives = []
non_all_negatives = []

for label_file in label_files:
    file_path = os.path.join(src_labels_dir, label_file)
    dataset = gdal.Open(file_path)
    array = dataset.ReadAsArray()
    if (array == 0).all():
        all_negatives.append(label_file)
    else:
        non_all_negatives.append(label_file)

# 输出统计信息
print(f"全负样本数量: {len(all_negatives)}")
print(f"非全负样本数量: {len(non_all_negatives)}")

# 创建新的文件排序
interleaved_files = []
non_neg_index = 0
neg_index = 0
while non_neg_index < len(non_all_negatives) and neg_index < len(all_negatives):
    # 每两个非全负样本后插入一个全负样本
    if non_neg_index + 2 <= len(non_all_negatives):
        interleaved_files.extend(non_all_negatives[non_neg_index:non_neg_index+2])
        non_neg_index += 2
    else:
        interleaved_files.append(non_all_negatives[non_neg_index])
        non_neg_index += 1
    interleaved_files.append(all_negatives[neg_index])
    neg_index += 1

# 添加剩余的非全负样本
interleaved_files.extend(non_all_negatives[non_neg_index:])

# 重命名和复制文件
for i, filename in enumerate(interleaved_files):
    new_name = f"{i+1}.tif"  # 更新这里，使文件名从1.tif开始
    # 复制训练影像数据
    original_image_path = os.path.join(src_images_dir, filename.replace("_label", ""))
    new_image_path = os.path.join(dst_images_dir, new_name)
    copy2(original_image_path, new_image_path)
    # 复制标签文件
    original_label_path = os.path.join(src_labels_dir, filename)
    new_label_path = os.path.join(dst_labels_dir, new_name)
    copy2(original_label_path, new_label_path)
