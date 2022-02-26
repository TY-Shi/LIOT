###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import torch
import torch.nn as nn
from torch.utils import data
#from unet import UNet
from iternet import Iternet
import argparse
import sys
import os

torch.cuda.set_device(0)
from topoloss_fast_pytorch import *

sys.path.insert(0, './lib/')
from help_functions import *
from torch_augment import random_augmentation
# function to obtain data for training/testing (validation)
from extract_patches import get_data_training
from crop_dataset import CurvilinearDataset,CurvilinearDataset_val


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    batch_num = len(train_loader)
    print('Train_Epoch: [{}] current learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    loss_l = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = random_augmentation(inputs,targets)
        inputs = inputs.cuda()
        targets = targets.cuda()
        net_out = model(inputs)  # bs, 2, H, W
        bs, n_class, h, w = net_out.size()
        net_out = net_out.view(bs, n_class, h * w)
        # net_out = net_out.permute(0, 2, 1)
        targets = targets.view(bs, -1)
        loss = criterion(net_out, targets)
        loss_l.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            info = 'Train_Epoch: [{}][{}/{}]\tLoss:{:.5f}'.format(epoch, batch_idx, batch_num, loss.item())
            #print(info)

    return np.mean(loss_l)

def traintopo_epoch(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    batch_num = len(train_loader)
    print('Train_Epoch: [{}] current learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    loss_l = []
    topo_l = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        topoLoss = 0
        lamda = 0.005 #DRIVE
        inputs, targets = random_augmentation(inputs, targets)
        inputs = inputs.cuda()
        targets = targets.cuda()
        net_out = model(inputs)  # bs, 2, H, W
        bs, n_class, h, w = net_out.size()
        net_out_ce = net_out.view(bs, n_class, h * w)
        # net_out = net_out.permute(0, 2, 1)
        targets_ce = targets.view(bs, -1)
        softmax = nn.Softmax2d()
        likelihoodMap = softmax(net_out)[:, 1, :, :]
        #pred_class = torch.argmax(net_out, dim=1).float()
        length = 0
        for i in range(len(likelihoodMap)):
            if torch.min(targets[i,0])==0 and torch.max(targets[i,0])==1:
                loss_topo = getTopoLoss(likelihoodMap[i], targets[i,0])
                topoLoss += loss_topo
                length += 1
        topoLoss = topoLoss / length
        loss = criterion(net_out_ce, targets_ce) + lamda * topoLoss
        loss_l.append(loss.item())
        topo_l.append((lamda * topoLoss).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            info = 'Train_Epoch: [{}][{}/{}]\tLoss:{:.5f}'.format(epoch, batch_idx, batch_num, loss.item())
            #print(info)

    return np.mean(loss_l), np.mean(topo_l)

def validtopo_epoch(valid_loader, model, criterion, device):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    #batch_num = len(valid_loader)
    loss_l = []
    topo_l = []
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        topoLoss = 0
        lamda = 0.005# DRIVE
        inputs, targets = random_augmentation(inputs, targets)
        inputs = inputs.cuda()
        targets = targets.cuda()
        net_out = model(inputs)  # bs, 2, H, W
        bs, n_class, h, w = net_out.size()
        net_out_ce = net_out.view(bs, n_class, h * w)
        targets_ce = targets.view(bs, -1)
        softmax = nn.Softmax2d()
        likelihoodMap = softmax(net_out)[:, 1, :, :]
        length = 0
        for i in range(len(likelihoodMap)):
            if torch.min(targets[i, 0]) == 0 and torch.max(targets[i, 0]) == 1:
                loss_topo = getTopoLoss(likelihoodMap[i], targets[i,0])
                topoLoss += loss_topo
                length += 1
        topoLoss = topoLoss / length
        loss = criterion(net_out_ce, targets_ce) + lamda * topoLoss
        loss_l.append(loss.item())
        topo_l.append((lamda * topoLoss).item())
    return np.mean(loss_l), np.mean(topo_l)

def valid_epoch(valid_loader, model, criterion):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    loss_l = []
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        inputs, targets = random_augmentation(inputs, targets)
        inputs = inputs.cuda()
        targets = targets.cuda()
        net_out = model(inputs)  # bs, 2, H, W
        bs, n_class, h, w = net_out.size()
        net_out = net_out.view(bs, n_class, h * w)
        targets = targets.view(bs, -1)
        loss = criterion(net_out, targets)
        loss_l.append(loss.item())

    return np.mean(loss_l)


# ========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')

#Data path
path_data = config.get('data paths', 'path_local')
# Experiment name
name_experiment = config.get('experiment name', 'name')
if not os.path.exists(name_experiment):
    os.mkdir(name_experiment)
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#val setting
val_batch_size = 12

# ============ Load the data and divided in patches
DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth')
DRIVE_val_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
DRIVE_val_groudTruth = path_data + config.get('data paths', 'test_groundTruth')
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

train_CurvilinearData = CurvilinearDataset(DRIVE_train_imgs_original, DRIVE_train_groudTruth, repeat=30, batch_size=32, patch_h=patch_height, patch_w=patch_width)
val_CurvilinearData = CurvilinearDataset_val(DRIVE_val_imgs_original, DRIVE_val_groudTruth,repeat = 1)
train_loader = data.DataLoader(train_CurvilinearData, batch_size=batch_size, pin_memory=True,shuffle=True)
valid_loader = data.DataLoader(val_CurvilinearData, batch_size=val_batch_size, pin_memory=True, shuffle=False)

# ============ Set model
model = Iternet(n_channels=4, n_classes=2).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if not os.path.exists(name_experiment + '/model'):
    os.mkdir(name_experiment + '/model')
best_val_loss = np.inf
best_top_loss = np.inf
pretrain_epoch = 150 #after pretrain use topo loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(0,N_epochs):
    if epoch<=pretrain_epoch:
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
        valid_loss = valid_epoch(valid_loader, model, criterion)
        if valid_loss < best_val_loss:
            torch.save(model.state_dict(), name_experiment + '/' + 'model/model_best.pth')
            best_val_loss = valid_loss
            print("=> Save Best")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), name_experiment + '/' + 'model/model{}.pth'.format(epoch))
        if epoch == pretrain_epoch:
            best_top_loss = np.inf

    else:
        train_loss, train_topoloss = traintopo_epoch(train_loader, model, criterion, optimizer, epoch, device)
        valid_loss, valid_topoloss = validtopo_epoch(valid_loader, model, criterion, device)
        print("Epoch{}\tTrainLoss: {:.5f}\tValLoss: {:.5f}".format(epoch, train_loss, valid_loss))
        print("Epoch{}\tTrainTopoLoss: {:.5f}\tValTopoLoss: {:.5f}".format(epoch, train_topoloss, valid_topoloss))
        if valid_loss < best_top_loss:
            torch.save(model.state_dict(), name_experiment + '/' + 'model/model_best_to.pth')
            best_top_loss = valid_loss
            print("=> Save Best")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), name_experiment + '/' + 'model/model_to_{}.pth'.format(epoch))