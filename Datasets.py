import torch
import torchvision.transforms as transforms

import os
from os.path import join
import glob
import cv2
import numpy as np
import pandas as pd
from math import floor

from PIL import Image
import torch.utils.data as data


# 这个类用于加载和处理微表情数据集，特别适用于处理包含不同数据集（如 CASME、SAMM、SMIC）和不同类别标签的情况。处理图像与便签的映射
#这可以LOSO，但是需要指定val_subject.所以后面需要一个循环制定.
class ME_dataset(data.Dataset):
    def __init__(self, transform=None, train=True, val_subject=1, dataset='smic', datadir='/home/ff/lyl-project/SLSTT-main/inputs', combined = False):
        self.transform = transform
        self.val_subject = val_subject
        self.train = train
        self.dataset = dataset
        self.data_folder = join(datadir,dataset,'LOF')
        self.labels = ('negative','positive','surprise')
        if combined:
            data_gt = pd.read_csv(join(datadir,'combined_3class_gt.csv'), header=None, names=['Dataset','Subject','Filename','Class'])
        elif dataset == 'casme2':
            # 根据数据集的不同，设置不同的标签列表。(只选择casme2中的5类来划分数据集)
            # self.labels = ('disgust', 'happiness', 'repression', 'surprise', 'others')
            self.labels = ('positive', 'negative', 'surprise')
        elif dataset == 'samm':
            self.labels = ('Other', 'Anger', 'Contempt', 'Happiness', 'Surprise')
        # elif dataset == 'ck+':
        #     self.labels = ('neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')
        #     self.ftype = '.png'
        

        # casme2银蛇成3类:::::::::定义原始的标签列表
        original_labels = ('happiness', 'surprise', 'disgust', 'sadness', 'fear', 'repression', 'others')


        # 加载数据集的编码信息
        data_info = pd.read_csv(join(datadir,dataset,'coding.csv'))
        # label indexing, {'negative': array(0}, ...}创建一个标签到索引的映射。label_index :  {'disgust': 0, 'happiness': 1, 'repression': 2, 'surprise': 3, 'others': 4}
        # self.label_index = {label : i for i, label in enumerate(self.labels)}      #定义为多个类时.
         # 创建标签到索引的映射
        self.label_index = {
            'positive': 0,  # 积极
            'negative': 1,  # 消极
            'surprise': 2   # 惊讶
        }
         # 创建原始标签到映射标签的映射
        self.original_to_mapped = {
            'happiness': 'positive',  # happiness 映射到 positive
            'disgust': 'negative',    # disgust 映射到 negative
            'repression': 'negative', # repression 映射到 negative
            'surprise': 'surprise',   # surprise 映射到 surprise
        }


        self.subject_num = data_info.Subject.nunique()
        if combined:
            self.subject_num = data_gt.Subject.nunique()
        image_list = glob.glob(join(self.data_folder,'*','*','*'+'.jpg'), recursive=True)
        if self.dataset == 'smic':
            image_list = glob.glob(join(self.data_folder,'*','*','*','*'+'.bmp'), recursive=True)
        
        self.train_image = image_list.copy()
        self.val_image = []
        self.val_label = {}
        self.val_images = {}
        self.train_images = {}
        self.train_label = {}
        val_imgs = {}
        train_imgs = {}


        self.train_image_paths = []
        self.train_labels = []
        self.val_image_paths = []
        self.val_labels = []
        
        # 循环遍历每个图像，根据数据集的不同，将图像分配到训练集或验证集，并为每个图像分配相应的标签。这个过程涉及到读取数据集的元数据，检查图像是否存在，并将图像路径和标签存储在相应的字典中。
        # 如果图像的 Emotion 标签在 self.labels 列表中，则使用 self.label_index 来获取该标签的索引，并将其存储在相应的 train_label 或 val_label 中。只获取那5类数据。
        # if self.dataset == 'casme2':
        #     for image in image_list:
        #         sub = data_info[data_info['Subject'].isin([int(image.split('/')[-3][3:])])]
        #         ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
        #         if combined:
        #             sub = data_gt[data_gt['Subject'].isin([image.split('/')[-3]])]
        #             ep = sub[sub['Filename'].isin([image.split('/')[-2]])]

        #         if sub.empty or ep.empty:
        #             self.train_image.remove(image)
        #         elif str(ep.Subject.values[0]) in [str(self.val_subject), 'sub'+str(self.val_subject).zfill(2)]:
        #             if combined:
        #                 self.val_label[image] = ep.Class.values[0]
        #                 if ep.Subject.values[0]+ep.Filename.values[0] not in val_imgs.keys():
        #                     val_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
        #                 val_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
        #             elif ep.Emotion.values[0] in self.labels:
        #                 self.val_label[image] = self.label_index[ep.Emotion.values[0]]
        #                 if 'sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0] not in val_imgs.keys():
        #                     val_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]] = []
        #                 val_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]].append(image)
        #             self.train_image.remove(image)
        #         else:
        #             if combined:
        #                 self.train_label[image] = ep.Class.values[0]
        #                 if ep.Subject.values[0]+ep.Filename.values[0] not in train_imgs.keys():
        #                     train_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
        #                 train_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
        #             elif ep.Emotion.values[0] in self.labels:
        #                 self.train_label[image] = self.label_index[ep.Emotion.values[0]]
        #                 if 'sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0] not in train_imgs.keys():
        #                     train_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]] = []
        #                 train_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]].append(image)
        #             else:
        #                 self.train_image.remove(image)

        # ----------------------这是处理casme2  3类情况的时候.  且这是处理4维的数据
        if self.dataset == 'casme2':
            for image in image_list:
                sub = data_info[data_info['Subject'].isin([int(image.split('/')[-3][3:])])]
                ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                if combined:
                    sub = data_gt[data_gt['Subject'].isin([image.split('/')[-3]])]
                    ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                if sub.empty or ep.empty:
                    continue
                original_emotion = ep.Emotion.values[0]
                if original_emotion not in self.original_to_mapped:
                    continue
                mapped_emotion = self.original_to_mapped[original_emotion]
                if str(ep.Subject.values[0]) in [str(self.val_subject), 'sub' + str(self.val_subject).zfill(2)]:
                    self.val_labels.append(self.label_index[mapped_emotion])  # 添加标签到列表
                    self.val_image_paths.append(image)  # 添加图像路径到列表
                else:
                    self.train_labels.append(self.label_index[mapped_emotion])  # 添加标签到列表
                    self.train_image_paths.append(image)  # 添加图像路径到列表

                
        elif self.dataset == 'samm':
            for image in image_list:
                    sub = data_info[data_info['Subject'].isin([int(image.split('/')[-3])])]
                    ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                    if combined:
                        sub = data_gt[data_gt['Subject'].isin([str(int(image.split('/')[-3]))])]
                        ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                    
                    if sub.empty or ep.empty:
                        self.train_image.remove(image)

                    elif str(ep.Subject.values[0]) in [str(self.val_subject), str(self.val_subject).zfill(3)]:
                        if combined:
                            self.val_label[image] = ep.Class.values[0]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in val_imgs.keys():
                                val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        elif ep.Emotion.values[0] in self.labels:
                            self.val_label[image] = self.label_index[ep.Emotion.values[0]]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in val_imgs.keys():
                                val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        self.train_image.remove(image)
                    else:
                        if combined:
                            self.train_label[image] = ep.Class.values[0]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in train_imgs.keys():
                                train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        elif ep.Emotion.values[0] in self.labels:
                            self.train_label[image] = self.label_index[ep.Emotion.values[0]]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in train_imgs.keys():
                                train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        else:
                            self.train_image.remove(image)
        elif self.dataset == 'smic':
            for image in image_list:
                sub = data_info[data_info['Subject'].isin([image.split('/')[-4]])]
                ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                if combined:
                    sub = data_gt[data_gt['Subject'].isin([image.split('/')[-4]])]
                    ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                
                if sub.empty or ep.empty:
                    self.train_image.remove(image)

                elif str(ep.Subject.values[0]) in [str(self.val_subject), 's'+str(self.val_subject)]:
                    if combined:
                        self.val_label[image] = ep.Class.values[0]
                        if ep.Subject.values[0]+ep.Filename.values[0] not in val_imgs.keys():
                            val_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
                        val_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
                    elif ep.Emotion.values[0] in self.labels:
                        self.val_label[image] = self.label_index[ep.Emotion.values[0]]
                        if str(ep.Subject.values[0])+ep.Filename.values[0] not in val_imgs.keys():
                            val_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]] = []
                        val_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]].append(image)
                    self.train_image.remove(image)
                else:
                    if combined:
                        self.train_label[image] = ep.Class.values[0]
                        if ep.Subject.values[0]+ep.Filename.values[0] not in train_imgs.keys():
                            train_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
                        train_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
                    elif ep.Emotion.values[0] in self.labels:
                        self.train_label[image] = self.label_index[ep.Emotion.values[0]]
                        if str(ep.Subject.values[0])+ep.Filename.values[0] not in train_imgs.keys():
                            train_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]] = []
                        train_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]].append(image)
                    else:
                        self.train_image.remove(image)
        # elif self.dataset == 'ck+':
        #     for image in image_list:
        #         sub = data_info[data_info['Subject'].isin([image.split('/')[-3]])]
        #         ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
        #         self.train_label[image] = self.label_index[ep.Emotion.values[0]]
        #         if ep.Subject.values[0]+str(ep.Filename.values[0]) not in train_imgs.keys():
        #             train_imgs[ep.Subject.values[0]+str(ep.Filename.values[0])] = []
        #         train_imgs[ep.Subject.values[0]+str(ep.Filename.values[0])].append(image)

        for i, key in enumerate(val_imgs.keys(), 0):
            self.val_images[i] = val_imgs[key]
        for i, key in enumerate(train_imgs.keys(), 0):
            self.train_images[i] = train_imgs[key]


    # 这个方法根据索引 index 获取一个数据项。它返回一个包含图像和标签的元组。
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, subject) where target is index of the target class, subject is the subject number
        """
        if self.train:
            image_path = self.train_image_paths[index]
            label = self.train_labels[index]
        else:
            image_path = self.val_image_paths[index]
            label = self.val_labels[index]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
       

    def __len__(self):
        if self.train:
            return len(self.train_image_paths)
        else:
            return len(self.val_image_paths)


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((28,28)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        # transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    ])
    # 创建数据集实例
    # dataset = ME_dataset(transform=data_transforms, train=True,val_subject=1, dataset='casme2', datadir='/home/ff/lyl-project/SLSTT-main/inputs', combined=False)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    # 创建训练集和测试集实例
    train_dataset = ME_dataset(transform=data_transforms, train=True, val_subject=1, dataset='casme2', datadir='/home/ff/lyl-project/SLSTT-main/inputs', combined=False)
    val_dataset = ME_dataset(transform=data_transforms, train=False, val_subject=1, dataset='casme2', datadir='/home/ff/lyl-project/SLSTT-main/inputs', combined=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# 打印 train_loader 中的每个批次
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")  # 打印数据的形状
        print(f"Target shape: {target.shape}")  # 打印标签的形状
        print(f"Target values: {target}")  # 打印标签的值
        print(len(train_loader))

    # 这样训练
    # subjects = data_info['Subject'].unique()  # 假设 data_info 是包含所有受试者信息的 DataFrame
    # for val_subject in subjects:
#       train_dataset = ME_dataset(transform=data_transforms, train=True, val_subject=val_subject, dataset='casme2', datadir='/home/ff/lyl-project/SLSTT-main/inputs', combined=False)
#       val_dataset = ME_dataset(transform=data_transforms, train=False, val_subject=val_subject, dataset='casme2', datadir='/home/ff/lyl-project/SLSTT-main/inputs', combined=False)
        
    #   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    #   val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
            # pass 




    # # 将验证集数据写入文本文件    这两部可以手动弄好实现loso
    # with open('sub'+str(val_subject).zfill(2)+'_val_data.txt', 'w') as f:
    #     for image_path, label in zip(val_dataset.val_image_paths, val_dataset.val_labels):
    #             f.write(f"{image_path} {label}\n")
    
    # # 将训练集数据写入文本文件
    # with open('sub'+str(val_subject).zfill(2)+'_train_data.txt', 'w') as f:
    #     for image_path, label in zip(train_dataset.train_image_paths, train_dataset.train_labels):
    #             f.write(f"{image_path} {label}\n")
    # print(f'{str(val_subject).zfill(2)}'+ " data have been saved ")