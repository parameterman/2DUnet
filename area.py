import argparse
from PIL import Image
import os

def calculate_white_pixel_area(image_path):
    # 打开图像
    with Image.open(image_path) as img:
        # 将图像转换为灰度图
        img_gray = img.convert('L')
        
        # 获取图像的尺寸
        width, height = img_gray.size
        
        # 初始化白色像素计数器
        white_pixel_count = 0
        
        # 遍历图像中的每个像素
        for x in range(width):
            for y in range(height):
                # 检查像素是否为白色（灰度值为255）
                if img_gray.getpixel((x, y)) == 255:
                    white_pixel_count += 1
    
    return white_pixel_count

if __name__ == '__main__':
# 图像文件夹路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1',default=None)
    parser.add_argument('--file2',default=None)
    args = parser.parse_args()

    if args.file1:
        before = args.file1
    else:
        raise ValueError("Please input the before dir path")
    if args.file2:
        after = args.file2
    else:
        raise ValueError("Please input the after dir path")
        # print(before)
        # print(after)
    image_folder_path = 'dataset\\predictions'
    folder_1 = os.path.join(image_folder_path, before)
    folder_2 = os.path.join(image_folder_path, after)
    # 获取文件夹中所有图像文件的路径
    image_files_1 = [os.path.join(folder_1, f) for f in os.listdir(folder_1) if f.endswith(('.png'))]
    image_files_2 = [os.path.join(folder_2, f) for f in os.listdir(folder_2) if f.endswith(('.png'))]
    # 计算每个图像中白色像素的面积
    areas_1 = 0
    areas_2 = 0
    for image_file in image_files_1:
        white_area = calculate_white_pixel_area(image_file)
        areas_1 += white_area
    for image_file in image_files_2:
        white_area = calculate_white_pixel_area(image_file)
        areas_2 += white_area
    # 输出结果
    print("Before: ", areas_1)
    print("After: ", areas_2)