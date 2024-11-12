import os
import glob
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path #dataset/images
        self.dir_list = os.listdir(self.data_path)  #dataset/images/*
        print(self.dir_list)
        self.img_list = []
        for dir in self.dir_list:
            print(dir)
            self.img_list =  self.img_list + glob.glob(os.path.join(self.data_path, dir, '*.png'))  #dataset/images/xxxx/*.png
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.img_list[index]
        print(image_path)
        # 根据image_path生成label_path
        label_path = image_path.replace('images', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (534, 534))
        label = cv2.resize(label, (534, 534))
        # img.reshape(1,534,534)
        # label.reshape(1,534,534)
        print(image.shape)
        print(label.shape)
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label
    def __len__(self):
        return len(self.img_list)

def create_dataset():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        os.makedirs('dataset/images')
        os.makedirs('dataset/label')

    # x_train = np.array( [] )
    x_train = []
    # x_label = np.array( [] )
    x_label = []
    target_size = (534,534,4)
    target_size_label = (534,534)
    for file in glob.glob('D:\\OS\\Unet\\dataset\\images\\*'):
        print(file)
        for file_name in glob.glob(file+'\\*'):
            #
            label_path = file_name.replace('images','label')
            image_path = file_name
            #读取训练图片和标签
            img = cv2.imread(image_path)
            label = cv2.imread(label_path)
            #将数据转为单通道
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            #将数据缩放为534x534
            img = cv2.resize(img, (534, 534))
            label = cv2.resize(label, (534, 534))
            img.reshape(1,534,534)
            label.reshape(1,534,534)
            if label.max() > 1:
                label = label / 255.0
            flipCode = random.choice([-1,0,1,2])
            if flipCode != 2:
                img = cv2.flip(img, flipCode)
                label = cv2.flip(label, flipCode)
            
    #         img = np.array(Image.open(file_name), dtype='float32') /255.0
    #         pad_width = []
    #         next = False
    #         # print(img.shape)
    #         for origin_dim,target_dim in zip(img.shape,target_size):
    #             if origin_dim < target_dim:
    #                 row_pad = int(0.5*(target_dim-origin_dim))
    #                 col_pad = int(0.5*(target_dim-origin_dim))
    #                 if(target_dim - origin_dim - row_pad - col_pad) ==1:
    #                     print("pad over 1")
    #                     col_pad += 1
    #                 elif(target_dim - origin_dim - row_pad - col_pad) ==2:
    #                     print("pad over 2")
    #                     col_pad += 1
    #                     row_pad += 1
    #                 pad_width.append((row_pad,col_pad))
                    
    #             elif origin_dim > target_dim:
    #                 next = True
    #             else:
    #                 pad_width.append((0,0))
            
            
    #         if not next:
    #             img = np.pad(img,pad_width=pad_width,mode='constant',constant_values=0)
    #             if img.shape[0] != 534 or img.shape[1] != 534:
    #                 print(file_name)
    #                 print(img.shape)
    #                 raise ValueError('Image size is not 534x534')
    #             # image = cv2.imread(image_path)
    #             img = np.transpose(img,(2,0,1))

    #             img = img.reshape((1,)+img.shape)
    #             x_train.append(img)
    #             print(img.shape)
    #         # print(x_train)
    #         # x_train = np.append(x_train, img)
    
    # for file in glob.glob('.\\dataset\\label\\*'):
    #     print(file)
    #     for file_name in glob.glob(file+'\\*'):
            
    #         img = np.array(Image.open(file_name), dtype='float32') /255.0
    #         pad_width = []
    #         next = False
    #         # print(img.shape)
    #         for origin_dim,target_dim in zip(img.shape,target_size_label):
    #             if origin_dim < target_dim:
    #                 row_pad = int(0.5*(target_dim-origin_dim))
    #                 col_pad = int(0.5*(target_dim-origin_dim))
    #                 if(target_dim - origin_dim - row_pad - col_pad) ==1:
    #                     print("pad over 1")
    #                     col_pad += 1
    #                 elif(target_dim - origin_dim - row_pad - col_pad) ==2:
    #                     print("pad over 2")
    #                     col_pad += 1
    #                     row_pad += 1
    #                 pad_width.append((row_pad,col_pad))
    #             elif origin_dim > target_dim:
    #                 next = True
    #             else:
    #                 pad_width.append((0,0))
            
    #         if not next:
    #             img = np.pad(img,pad_width=pad_width,mode='constant',constant_values=0)
    #             if img.shape[0] != 534 or img.shape[1] != 534:
    #                 print(file_name)
    #                 print(img.shape)
    #                 raise ValueError('Image size is not 534x534')
    #             # img = np.transpose(img,(2,0,1))
    #             img = img.reshape((1,)+img.shape)
    #             img = img.reshape((1,)+img.shape)
    #             x_label.append(img)
    #             print(img.shape)

    # np.random.seed(116)
    # x_train = np.array(x_train)
    
    # x_label = np.array(x_label)
    # np.random.shuffle(x_train)
    # print(x_train.shape)
    # np.random.seed(116)
    # np.random.shuffle(x_label)
    # # print(x_label.shape)

    np.save('dataset\\x_train.npy', x_train[:4800])
    np.save('dataset\\x_label.npy', x_label[:4800])
    np.save('dataset\\x_test.npy', x_train[4800:])
    np.save('dataset\\x_test.npy', x_label[4800:])
if __name__ == '__main__':
    create_dataset()
