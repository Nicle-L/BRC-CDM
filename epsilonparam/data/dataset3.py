import os
import torch
import logging

from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random

#文件夹编号/图像编号/图像文件名
import os
import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio

class DataSet(data.Dataset):
    def __init__(self, path="E:/ljh/test3/data/train.txt", im_height=256, im_width=256):
        self.file_list, self.ref_list = self.load_file_list(path)  # 保存文件列表和参考图像列表
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.file_list))

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "E:/ljh/test3/data2"
        file_list = []
        ref_list = []
        for line in lines:
            line = line.strip()  # 移除换行符
            parts = line.split('/')  # 分割文件路径
            if len(parts) == 3:
                folder, sub_folder, filename_with_extension = parts
                filename = os.path.splitext(filename_with_extension)[0]  # 移除扩展名
                mat_file = os.path.join(root_dir, folder, sub_folder, filename_with_extension)
                file_list.append(mat_file)

                # 修改参考图像文件名
                ref_filename = str(int(filename) - 1) + '.mat'
                ref_mat_file = os.path.join(root_dir, folder, sub_folder, ref_filename)
                ref_list.append(ref_mat_file)

        return file_list, ref_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        ref_file = self.ref_list[index]  # 获取对应的参考图像文件路径

        # 读取.mat文件
        input_data = sio.loadmat(mat_file)
        ref_data = sio.loadmat(ref_file)

        # 从.mat文件中提取图像数据
        input_image = input_data['data']
        ref_image = ref_data['data']

        # 数据处理
        input_image_data = torch.tensor(input_image.astype(np.float32))
        input_image_data = input_image_data.permute(2, 0, 1)

        ref_image_data = torch.tensor(ref_image.astype(np.float32))
        ref_image_data = ref_image_data.permute(2, 0, 1)



        return input_image_data, ref_image_data

class testDataSet(data.Dataset):
    def __init__(self, path="E:/ljh/test3/data/test.txt", im_height=256, im_width=256):
        self.file_list, self.ref_list = self.load_file_list(path)  # 保存文件列表和参考图像列表
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.file_list))

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "E:/ljh/test3/data2"
        file_list = []
        ref_list = []
        for line in lines:
            line = line.strip()  # 移除换行符
            parts = line.split('/')  # 分割文件路径
            if len(parts) == 3:
                folder, sub_folder, filename_with_extension = parts
                filename = os.path.splitext(filename_with_extension)[0]  # 移除扩展名
                mat_file = os.path.join(root_dir, folder, sub_folder, filename_with_extension)
                file_list.append(mat_file)

                # 修改参考图像文件名
                ref_filename = str(int(filename) - 1) + '.mat'
                ref_mat_file = os.path.join(root_dir, folder, sub_folder, ref_filename)
                ref_list.append(ref_mat_file)

        return file_list, ref_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        ref_file = self.ref_list[index]  # 获取对应的参考图像文件路径

        # 读取.mat文件
        input_data = sio.loadmat(mat_file)
        ref_data = sio.loadmat(ref_file)

        # 从.mat文件中提取图像数据
        input_image = input_data['data']
        ref_image = ref_data['data']

        # 数据处理
        input_image_data = torch.tensor(input_image.astype(np.float32))
        input_image_data = input_image_data.permute(2, 0, 1)

        ref_image_data = torch.tensor(ref_image.astype(np.float32))
        ref_image_data = ref_image_data.permute(2, 0, 1)



        return input_image_data, ref_image_data


