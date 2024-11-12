import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path #dataset/images
        self.dir_list = os.listdir(self.data_path)  #dataset/images/*
        # print(self.dir_list)
        self.img_list = []
        for dir in self.dir_list:
            
            if int(dir) < 220:
                # print(dir)
                self.img_list =  self.img_list + glob.glob(os.path.join(self.data_path, dir, '*.png'))  #dataset/images/xxxx/*.png
        # glob.glob(os.path.join(self.data_path, 'images')) # 读取所有的目录

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.img_list[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('images', 'label')
        # print(label_path)
        # print(image_path)
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (534, 534))
        label = cv2.resize(label, (534, 534))
        
        # print(image.shape)
        # print(label)
        # 处理标签，将像素值为255的改为1
       
        _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
        if label.max() > 1:
            label = label / 255
        # cv2.imwrite("test.png", label)
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1,534,534)

        label = label.reshape(1,534,534)
        
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.img_list)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("dataset\\images")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=5, 
                                               shuffle=False)
    # for image, label in train_loader:
    #     print(image.shape)