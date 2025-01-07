import os
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from PIL import Image
import torchvision.transforms as ts
from torch.utils.data.dataset import Dataset
import random
import numbers
import torchvision.transforms.functional as TF


def calculate_means_and_stds(data):
    # 初始化三个通道的总和和数量计数器
    channel_sums = np.zeros(3)
    channel_counts = 0

    # 遍历列表中的每个字典
    for item in data:
        image = item['image']  # 获取图像数据
        channel_sums += image.sum(axis=(1, 2))  # 对每个通道求和
        channel_counts += np.prod(image.shape[1:])  # 计算像素总数

    # 计算每个通道的均值
    channel_means = channel_sums / channel_counts

    # 初始化三个通道的平方差总和
    channel_squared_sums = np.zeros(3)

    # 再次遍历列表，计算平方差
    for item in data:
        image = item['image']
        channel_squared_sums += ((image - channel_means.reshape(3, 1, 1)) ** 2).sum(axis=(1, 2))

    # 计算每个通道的标准差
    channel_stds = np.sqrt(channel_squared_sums / channel_counts)
    print(channel_means, channel_stds)

    return channel_means, channel_stds

def transform(sample, mode):
    image = sample['image']
    label = sample['label']

    image = image.resize((256, 256))  # 调整大小
    label = label.resize((256, 256))
    label = label.convert('L')  # 从RGB变为灰度图, shape:（H, W)

    image = np.array(image).transpose(2, 0, 1)  # 把channel放最前面
    label = np.array(label)

    image = image / 255.0
    label = label / 255.0
    label = np.where(label > 0.5, 1, 0)

    if mode == 'train':
        pass
        # image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
    else:
        pass

    image = torch.tensor(image)
    mean = torch.tensor([0.55718692, 0.32169743, 0.23582221])
    std = torch.tensor([0.31797796, 0.22119849, 0.186987])
    image = (image - mean[:, None, None]) / std[:, None, None]
    label = torch.tensor(label)

    return {'image': image, 'label': label}

def randomflip_rotate(img, lab, p=0.5, degrees=0):
    if random.random() < p:
        img = TF.hflip(img)
        lab = TF.hflip(lab)
    if random.random() < p:
        img = TF.vflip(img)
        lab = TF.vflip(lab)

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        degrees = degrees
    angle = random.uniform(degrees[0], degrees[1])
    img = TF.rotate(img, angle)
    lab = TF.rotate(lab, angle)

    return img, lab

class Kvasir_SEG_dataset(Dataset):
    def __init__(self, folder='Kvasir-SEG', mode='train', transform=transform, start=0, end=199):
        self.transform = transform
        self.mode = mode
        self.folder = folder
        self.images = []
        self.masks = []
        # self.transformed_data = []

        if self.mode in ['train', 'validation', 'test']:
            # this is for cross validation
            # 遍历images文件夹
            images_dir = os.path.join(self.folder, 'images')
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(".jpg"):
                    img_path = os.path.join(images_dir, filename)
                    # img = Image.open(img_path)
                    # 将图片转换为numpy ndarray
                    # img = np.array(img).transpose(2, 0, 1)
                    self.images.append(img_path)
                    # print(np.array(img).shape)

            # 遍历masks文件夹
            masks_dir = os.path.join(self.folder, 'masks')
            for filename in os.listdir(masks_dir):
                if filename.lower().endswith(".jpg"):
                    mask_path = os.path.join(masks_dir, filename)
                    # mask = Image.open(mask_path)
                    # 将图片转换为numpy ndarray
                    # mask = np.array(mask).transpose(2, 0, 1)
                    self.masks.append(mask_path)
                    # print(np.array(mask).shape)

            # for image, mask in zip(self.images, self.masks):
            #     data = {'image': image, 'label': mask}
            #     self.transformed_data.append(self.transform(data, self.mode))


        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.images) == len(self.masks)
        self.images = self.images[start: end]
        self.masks = self.masks[start: end]

    def __getitem__(self, item: int):
        image = Image.open(self.images[item])
        mask = Image.open(self.masks[item])
        data = {'image': image, 'label': mask}
        sample = self.transform(data, self.mode)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = Kvasir_SEG_dataset()
    # calculate_means_and_stds(dataset.transformed_data)
