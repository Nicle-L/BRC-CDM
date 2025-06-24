import os
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DataSet1(Dataset):
    def __init__(self, path="E:/ljh/video-test3/data/train.txt", im_height=256, im_width=256):
        self.file_list, self.ref_list = self.load_file_list(path)
        self.im_height = im_height
        self.im_width = im_width
        print("dataset find image: ", len(self.file_list))

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
        root_dir = "E:/ljh/video-test3/data5"
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
            #transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()
        ])

    def load_file_list(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()

        root_dir = "E:/ljh/video-test3\data"
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

