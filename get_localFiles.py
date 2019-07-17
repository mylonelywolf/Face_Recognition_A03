import json
import os

import cv2
import numpy as np
from public_data import IMAGE_SIZE,JSON_PATH


class FileOperator:
    def __init__(self):
        # 读取训练数据
        self.images = []
        self.labels = []
        self.face_num = None

    def resize_image(self, image, height=IMAGE_SIZE, width=IMAGE_SIZE):
        """按照指定图像大小调整尺寸"""
        top, bottom, left, right = (0, 0, 0, 0)

        # 获取图像尺寸
        h, w, _ = image.shape  # (237, 237, 3)

        # 对于长宽不相等的图片，找到最长的一边
        longest = max(h, w)

        # 计算短边需要增加多上像素宽度使其与长边等长
        if h < longest:
            dh = longest - h
            top = dh // 2
            bottom = dh - top
        elif w < longest:
            dw = longest - w
            left = dw // 2
            right = dw - left
        else:
            pass

        # 填充边界所用的颜色
        border_color = [0, 0, 0]

        # 将当前图片填充成正方形
        # cv2.BORDER_CONSTANT表示填充颜色由value的值指定
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

        # 将图片缩放为指定大小
        return cv2.resize(constant, (height, width))

    def read_path(self, images_path):
        """从指定文件夹读取其下所有jpg格式图片以及其对应的标签名"""
        # self.images.clear()
        # self.labels.clear()
        # self.face_num = None
        for dir_item in os.listdir(images_path):

            # 从初始路径开始叠加，合并成可识别的操作路径
            full_path = os.path.abspath(os.path.join(images_path, dir_item))

            if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
                self.read_path(full_path)
            else:  # 文件
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path)
                    image = self.resize_image(image, IMAGE_SIZE, IMAGE_SIZE)

                    self.images.append(image)
                    self.labels.append(images_path.split('\\')[-1])

        return self.images, self.labels

    def load_dataset(self, dataset_path):
        """从指定路径读取训练数据"""
        # self.images, self.labels = self.read_path(dataset_path)
        self.read_path(dataset_path)
        print('labels:', self.labels)

        self.images = np.array(self.images)
        print(self.images.shape)

        # 转为集合去除重复项
        labels1 = list(set(self.labels))
        self.face_num = len(labels1)
        print('face_num:', self.face_num)
        name_list_dict = {}
        for i in range(self.face_num):
            name_list_dict[i] = labels1[i]
        with open(JSON_PATH, 'w') as f:
            f.write(json.dumps(name_list_dict))
        # print('name_list_dict:', name_list_dict)
        for index, name in name_list_dict.items():
            for i in range(len(self.labels)):
                if self.labels[i] == name:
                    self.labels[i] = index
        # print(labels)
        self.labels = np.array(self.labels)

        return self.images, self.labels, self.face_num


if __name__ == '__main__':
    fo = FileOperator()
    images, labels, face_num = fo.load_dataset("./data")
    print(face_num)
