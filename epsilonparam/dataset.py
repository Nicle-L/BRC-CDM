
from PIL.Image import Image

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms


class DataSet1(Dataset):
    def __init__(self, path="E:/ljh/video-test1/data/save/train.txt", im_height=256, im_width=256):
        self.file_list, self.ref_list = self.load_file_list(path)  # 保存文件列表和参考图像列表
        self.im_height = im_height
        self.im_width = im_width

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()
        ])

        print("dataset find image: ", len(self.file_list))

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "E:/ljh/video-test3/data"
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

        # 定义一个小的常数以避免除零
        epsilon = 1e-8

        # 检查最大值和最小值，并避免除以零
        input_min, input_max = np.min(input_image), np.max(input_image)
        ref_min, ref_max = np.min(ref_image), np.max(ref_image)

        # 对图像进行归一化处理，并在最大值和最小值相等时添加 epsilon
        input_image = (input_image - input_min) / (input_max - input_min + epsilon)
        ref_image = (ref_image - ref_min) / (ref_max - ref_min + epsilon)

        # 转换为uint8类型
        input_image = (input_image * 255).astype('uint8')
        ref_image = (ref_image * 255).astype('uint8')

        # 转换为PIL图像以便应用transforms
        input_image_pil = Image.fromarray(input_image)
        ref_image_pil = Image.fromarray(ref_image)

        # 应用数据增强转换
        input_image_transformed = self.transform(input_image_pil)
        ref_image_transformed = self.transform(ref_image_pil)

        return input_image_transformed, ref_image_transformed




import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image
import os

class testDataSet1(Dataset):
    def __init__(self, path="E:/ljh/video-test3/data/test.txt", im_height=256, im_width=256):
        self.file_list, self.ref_list = self.load_file_list(path)  # 保存文件列表和参考图像列表
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.file_list))

        self.transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(10),
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()
        ])

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "E:/ljh/video-test3/data"
        file_list = []
        ref_list = []
        for line in lines:
            line = line.strip()
            parts = line.split('/')
            if len(parts) == 3:
                folder, sub_folder, filename_with_extension = parts
                filename = os.path.splitext(filename_with_extension)[0]
                mat_file = os.path.join(root_dir, folder, sub_folder, filename_with_extension)
                file_list.append(mat_file)

                ref_filename = str(int(filename) - 1) + '.mat'
                ref_mat_file = os.path.join(root_dir, folder, sub_folder, ref_filename)
                ref_list.append(ref_mat_file)

        return file_list, ref_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        ref_file = self.ref_list[index]

        input_data = sio.loadmat(mat_file)
        ref_data = sio.loadmat(ref_file)

        input_image = input_data['data']
        ref_image = ref_data['data']

        epsilon = 1e-8

        input_min, input_max = np.min(input_image), np.max(input_image)
        ref_min, ref_max = np.min(ref_image), np.max(ref_image)

        input_image = (input_image - input_min) / (input_max - input_min + epsilon)
        ref_image = (ref_image - ref_min) / (ref_max - ref_min + epsilon)

        input_image = (input_image * 255).astype('uint8')
        ref_image = (ref_image * 255).astype('uint8')

        input_image_pil = Image.fromarray(input_image)
        ref_image_pil = Image.fromarray(ref_image)


        input_image_transformed = self.transform(input_image_pil)
        ref_image_transformed = self.transform(ref_image_pil)

        return input_image_transformed, ref_image_transformed



class valDataSet1(Dataset):
    def __init__(self, path="E:/ljh/video-test3/data/save/test.txt", im_height=256, im_width=256):
        self.file_list, self.ref_list = self.load_file_list(path)
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.file_list))

        self.transform = transforms.Compose([
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()
        ])

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "E:/ljh/compare2.0/data_class"
        file_list = []
        ref_list = []

        for line in lines:
            line = line.strip()
            mat_file = os.path.join(root_dir, line)
            file_list.append(mat_file)

            filename_with_extension = os.path.basename(line)
            filename = os.path.splitext(filename_with_extension)[0]
            ref_filename = self.get_ref_filename(filename)
            ref_mat_file = os.path.join(os.path.dirname(mat_file), ref_filename)
            ref_list.append(ref_mat_file)

        return file_list, ref_list

    def get_ref_filename(self, filename):
        if filename.isdigit():
            ref_filename = str(int(filename) - 1) + '.mat'
        else:
            if '_step_' in filename:
                base_name = filename.split('_step_')[0]
                ref_filename = base_name + '_step_1.mat'
            else:
                ref_filename = filename + '.mat'

        return ref_filename

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        ref_file = self.ref_list[index]

        input_data = sio.loadmat(mat_file)
        ref_data = sio.loadmat(ref_file)


        input_image = input_data['data']
        ref_image = ref_data['data']

        epsilon = 1e-8

        input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image) + epsilon)
        ref_image = (ref_image - np.min(ref_image)) / (np.max(ref_image) - np.min(ref_image) + epsilon)


        input_image = (input_image * 255).astype('uint8')
        ref_image = (ref_image * 255).astype('uint8')

        input_image_pil = Image.fromarray(input_image)
        ref_image_pil = Image.fromarray(ref_image)


        input_image_transformed = self.transform(input_image_pil)
        ref_image_transformed = self.transform(ref_image_pil)

        return input_image_transformed, ref_image_transformed

