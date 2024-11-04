from ntpath import join
from os import path
import os
import numpy as np
import cv2 
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
from tqdm import tqdm
import datetime
import torchvision.transforms as transforms
from Datasets import ME_dataset
import pandas as pd


class STSTNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(STSTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(5)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=5*5*16, out_features=out_channels)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.maxpool(x1)
        x1 = self.dropout(x1)
        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x2 = self.maxpool(x2)
        x2 = self.dropout(x2)
        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        x3 = self.maxpool(x3)
        x3 = self.dropout(x3)
        x = torch.cat((x1, x2, x3),1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def reset_weights(m): # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
#             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP) / (2*TP + FP + FN)
    num_samples = len([x for x in gt if x==1])
    if num_samples == 0:
        average_recall = np.nan
    else:
        average_recall = TP / num_samples
    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2 }
    
    #Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x==emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x==emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''





def predict():
    import torchvision
    model = STSTNet()
    weight_path = '../STSTNet-master/new_STSTNet_Weights/sub09.pth'
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = STSTNet().to(device)
    model.load_state_dict(torch.load(weight_path))
    # print(model)

    img = cv2.imread('../STSTNet-master/1234.png')
    print(img.dtype)
    # image = torchvision.transforms.ToTensor()
    # img = image(img)
    # print(img.dtype,img.shape)



def main(config):
    learning_rate = 0.00005    #0.00005
    batch_size = 256       #256
    epochs = 800            
        
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')



    if(config.train):
        if not path.exists('new_STSTNet_Weights'):
            os.mkdir('new_STSTNet_Weights')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
        
    t = time.time()


   # 获取当前日期
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # 创建一个基于当前日期的文件名
    file_path = f"logfile_{current_date}.txt"

    datadir='/home/ff/lyl-project/MyTest'
    dataset = '/datasets/casme2'
    data_info = pd.read_csv('/home/ff/lyl-project/MyTest/datasets/casme2/coding.csv')



    data_transforms = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    subjects = data_info['Subject'].unique()  # 假设 data_info 是包含所有受试者信息的 DataFrame
    for val_subject in subjects:

        train_dataset = ME_dataset(transform=data_transforms, train=True, val_subject=val_subject, dataset='/datasets/casme2', datadir='/home/ff/lyl-project/MyTest', combined=False)
        val_dataset = ME_dataset(transform=data_transforms, train=False, val_subject=val_subject, dataset='/datasets/casme2', datadir='/home/ff/lyl-project/MyTest', combined=False)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        weight_path = 'new_STSTNet_Weights' + '/' + 'sub'+ str(val_subject) + '.pth'


        # Reset or load model weigts
        model = STSTNet().to(device)
        if(config.train):
            model.apply(reset_weights)
        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            if(config.train):
                # Training
                model.train()
                train_loss         = 0.0
                num_train_correct  = 0
                num_train_examples = 0

                for batch_idx, (x, y) in enumerate(train_loader):
                    
                    optimizer.zero_grad()
                    x    = x.to(device)       #batch_size, out_channel, H, W
                    y    = y.to(device)       #batch_size大小的列表，应该是存储的每一个对应光流的标签值
                    yhat= model(x)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss         += loss.data.item() * x.size(0)
                    num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc   = num_train_correct / num_train_examples
                train_loss  = train_loss / len(train_loader)

            # Testing
            model.eval()
            val_loss       = 0.0
            num_val_correct  = 0
            num_val_examples = 0
            for batch in val_loader:
                x    = batch[0].to(device)
                y    = batch[1].to(device)
                yhat= model(x)
                # yhat = model(x)
                loss = loss_fn(yhat, y)

                val_loss         += loss.data.item() * x.size(0)
                # 正确样本累加数
                num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]
            # 正确数/验证总数量=准确率
            val_acc  = num_val_correct / num_val_examples
            # 计算平均损失
            val_loss = val_loss / len(val_loader)

            if  epoch % 100 == 0:            
                print('Epoch %3d/%3d, train loss: %5.4f, train acc: %5.4f, val loss: %5.4f, val acc: %5.4f' % (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        # Save Weights这是每训练完成+验证完成就保存权重文件，并没有选择最佳权重文件进行保存。
        if(config.train):
            torch.save(model.state_dict(), weight_path)

        #torch.max(yhat, 1)[1]-识别批中每个数据点概率最高的类的索引。并转换为列表
        print('Predicted    :', torch.max(yhat, 1)[1].tolist())
        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        #加入预测和真实列表，再次重新计算uf1和uar
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        # For UF1 and UAR computation
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))

        with open(file_path,'a') as  file:
            file.write('Subject:'+ str(val_subject)+ '     ' +'num_train_examples:'+ str(num_train_examples)+'  '+'num_val_examples:'+str(num_val_examples)+'\n')
            file.write('Predicted    :'+ str(torch.max(yhat, 1)[1].tolist())+'\n')
            file.write('Ground Truth :'+ str(y.tolist())+'\n')
            file.write(f'UF1:  {round(UF1, 4)} | UAR: {round(UAR, 4)} \n')
            file.write(f'train_acc:  {round(train_acc, 4)} | train_loss: {round(train_loss, 4)} \n')
            file.write(f'val_acc:  {round(val_acc, 4)} | val_loss: {round(val_loss, 4)} \n')
            file.write('######################################\n')


    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print('Total Time Taken:', time.time() - t)
    with open(file_path,'a') as  file:
        file.write('Final Evaluation:  '+'\n')
        file.write(f'UF1:  {round(UF1, 4)} | UAR: {round(UAR, 4)} \n')
        file.write('Total Time Taken:  '+ str(time.time() - t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train', type=strtobool, default=True) #Train or use pre-trained weight for prediction
    config = parser.parse_args()
    main(config)
    # predict()