import argparse
import glob
import numpy as np
import torch
import os
import cv2
from unet_build import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    begin_dir = input("请输入开始编号(221-270)：")
    end_dir = input("请输入结束编号(221-270)：")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device,weights_only=True))
    # 测试模式
    net.eval()
    full_acc = 0
    full_dice = 0
    full_iou = 0
    # 读取所有图片路径
    
    lenth = 0
    for dir_idx in range(int(begin_dir), int(end_dir)+1):
        dir_path = 'dataset\\images\\' + str(dir_idx).zfill(4) + '\\'
        tests_path = glob.glob(dir_path + '*.png')
        
        avg_acc = 0
        avg_dice = 0
        avg_iou = 0
        print("Dir:", dir_path)
        print("Num of test:", len(tests_path))
        if len(tests_path) == 0:
            continue
        # 遍历所有图片
        lenth += 1
        for test_path in tests_path:
            # 保存结果地址
            
            label_path = test_path.replace('images', 'label')
            save_res_path = test_path.replace('images', 'predictions')
            if not os.path.exists(os.path.dirname(save_res_path)):
                os.makedirs(os.path.dirname(save_res_path))
            print(save_res_path)
            save_res_path = save_res_path.split('.')[0] + '_res.png'
            # 读取图片
            img = cv2.imread(test_path)
            label = cv2.imread(label_path)
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            label = label.reshape(1, 1, label.shape[0], label.shape[1])
            
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            label_tensor = torch.from_numpy(label)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            
            
            # 预测
            pred = net(img_tensor)
            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            label = np.array(label_tensor.data.cpu()[0])[0]
            pred[pred > 0.1] = 255
            # 计算IOU  预测图像与标签图像的交集/并集
            iou = np.sum(np.logical_and(pred == 255, label == 255)) / np.sum(np.logical_or(pred == 255, label == 255))
            # print('IOU:', iou)
            avg_iou += iou
            # 计算dice系数  2*预测图像与标签图像的交集/两个图像的总和
            dice = 2 * np.sum(np.logical_and(pred == 255, label == 255)) / (np.sum(pred == 255) + np.sum(label == 255))
            # print('Dice:', dice)
            avg_dice += dice
            # 计算准确率  
            acc = np.sum(np.logical_and(pred == 255, label == 255)) / np.sum(label == 255)
            # print('Acc:', acc)
            avg_acc += acc
            # 计算平均值
        avg_acc /= len(tests_path)
        avg_dice /= len(tests_path)
        avg_iou /= len(tests_path)   
        print('Dir:', dir_idx, 'Avg Acc:', avg_acc, 'Avg Dice:', avg_dice, 'Avg IOU:', avg_iou)
        full_acc += avg_acc
        full_dice += avg_dice
        full_iou += avg_iou

        # 处理结果
        
        # pred[pred ] = 0
        # pred = (pred * 255).astype(np.uint8)
        
        # print(pred)

        # 保存图片
        cv2.imwrite(save_res_path, pred)
    
    print('Avg Acc 平均预测图像与标签图像的重合率:', full_acc / lenth)
    print('Avg Dice 平均相似度:', full_dice / lenth)
    print('Avg IOU 预测的分割区域和真实的分割区域之间的平均重叠程度:', full_dice / lenth)