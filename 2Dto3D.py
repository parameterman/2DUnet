import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os
# 假设所有的PNG图像切片都在同一个文件夹中，并且它们的命名是有序的
image_folder = 'dataset\\predictions\\0011'
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# 读取所有图像并转换为numpy数组
images = [np.array(plt.imread(os.path.join(image_folder, f))) for f in image_files]

# 假设所有图像的大小相同，提取第一个图像的大小
height, width = images[0].shape

# 创建一个空的3D数组来存储体素数据
# 这里使用8位深度，其中0表示空气，255表示物体
voxel_grid = np.zeros((len(images), height, width), dtype=np.uint8)

# 将每个图像切片转换为体素数据
for i, image in enumerate(images):
    # 这里简单地将图像的灰度值转换为二值化图像
    # 实际应用中可能需要更复杂的阈值处理
    voxel_grid[i, :, :] = (image > 128).astype(np.uint8) * 255

# 创建3D模型
mesh = trimesh.voxel.creation.voxelize(voxel_grid)

# 可视化3D模型
mesh.show()