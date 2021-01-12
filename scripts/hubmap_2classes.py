#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import h5py
import pandas as pd
import cv2
import tifffile
import skimage
from skimage import io, transform
from scipy import ndimage as nd
import json
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet import Unet
import sys
from sklearn.model_selection import GroupKFold
import time
from torch.nn import functional as F


# In[5]:


from albumentations import (
    Compose,
    OneOf,
    Flip,
    Rotate,
    RandomRotate90,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    GaussianBlur,
    GaussNoise,
    RandomCrop,
    ShiftScaleRotate,
    VerticalFlip,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    RandomScale,
    OpticalDistortion,
    ElasticTransform,
)


# In[6]:


BASE_PATH = "."
TRAIN_PATH = os.path.join(BASE_PATH, "train")
DATA_PATH = os.path.join(BASE_PATH,"train_tissue")

# tutorial can be found here:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
# class to load one pic
class KidneyFTUsDataset(Dataset):
    '''
    Kdiney with FTU mask dataset
    '''
    def __init__(self, root_dir, csv_file, mask_file,border_file, name_df = None, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with pic names.
            mask_file (string): hdf5 file containing masks 
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if name_df is not None:
            self.name_list = name_df
        else:
            self.name_list = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.mask_file = h5py.File(os.path.join(self.root_dir, mask_file), 'r')
        self.border_file = h5py.File(os.path.join(self.root_dir, border_file), 'r')
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.__transform_default()
            
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            tile_name = self.name_list.iloc[idx]['name']
            mask = self.mask_file[tile_name][()]*255
            mask = skimage.transform.resize(mask,(256,256))
            
            border = self.border_file[tile_name][()]*255
            border = skimage.transform.resize(border,(256,256))

            image = tifffile.imread(os.path.join(self.root_dir, 
                                                 tile_name + '.tiff'))
            image = cv2.resize(image,(256,256))

            enhanced_mask = np.array([mask,border])
            enhanced_mask = enhanced_mask.transpose(1,2,0)
            transformed = self.transform(image=image, mask=enhanced_mask)
            
            image = transformed['image']
            mask = transformed['mask']
            image = torch.Tensor(image).permute(2, 0, 1)
            mask = torch.Tensor(mask).permute(2, 0, 1)
            
            sample = {'image': image, 'mask': mask}
            return sample
        
    def __transform_default(self):
        return Compose([
            Normalize(max_pixel_value=255.0),
        ])
        


# In[8]:


aug_train = Compose([Flip(p=0.5),
                     RandomRotate90(p=0.5),
                     Rotate(limit=180, p=0.5),
                     Normalize(max_pixel_value=255.0),
        ])
aug_valid = Compose([Normalize(max_pixel_value=255.0)])


# # MODEL

# In[ ]:


class HuBMAP(torch.nn.Module):
    def __init__(self):
        super(HuBMAP, self).__init__()
        self.cnn_model = Unet('efficientnet-b2', encoder_weights="imagenet", classes=2, activation=None)
        #self.cnn_model.segmentation_head[0] = torch.nn.Conv2d(16, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.cnn_model.decoder.blocks.append(self.cnn_model.decoder.blocks[-1])
        #self.cnn_model.decoder.blocks[-2] = self.cnn_model.decoder.blocks[-3]
    
    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs


# In[9]:


# https://www.kaggle.com/vineeth1999/hubmap-pytorch-efficientunet-offline
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

class DiceBCELoss(torch.nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        input_mask = inputs[:,0,...]
        input_bord = inputs[:,1,...]
        input_mask = input_mask .contiguous().view(-1)
        input_bord = input_bord .contiguous().view(-1)
        
        
        target_mask = targets[:,0,...]
        target_bord = targets[:,1,...]
        target_mask = target_mask .contiguous().view(-1)
        target_bord = target_bord .contiguous().view(-1)
        
        intersection = (input_mask * target_mask).mean()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input_mask.mean() + target_mask.mean() + smooth)  
        BCE = F.binary_cross_entropy(input_bord, target_bord, reduction='mean')
        Dice_BCE = 0.1*BCE + 0.9*dice_loss
        
        return Dice_BCE.mean()    


# In[10]:


def HuBMAPLoss(images, targets, model, device):
    model.to(device)
    #print('transferring images to gpu')
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    criterion = DiceBCELoss()
    loss = criterion(outputs, targets)
    return loss, outputs


# In[11]:

def prepare_train_valid_dataloader(df, fold):
    train_ids = df.loc[~df.Folds.isin(fold)]
    val_ids = df.loc[df.Folds.isin(fold)]
    train_ds = KidneyFTUsDataset(root_dir = DATA_PATH, 
                                        csv_file = 'names.csv', 
                                        mask_file = 'mask.h5',
                                        border_file = 'borders.h5',
                                        name_df = train_ids,
                                        transform=aug_train)
    val_ds = KidneyFTUsDataset(root_dir = DATA_PATH, 
                                        csv_file = 'names.csv', 
                                        mask_file = 'mask.h5',
                                        border_file = 'borders.h5',
                                        name_df = val_ids,
                                        transform=aug_valid)
    train_loader = DataLoader(train_ds, batch_size=12, 
                              pin_memory=True, shuffle=True, 
                              num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=12, 
                            pin_memory=True, shuffle=False, 
                            num_workers=1)
    return train_loader, val_loader

def train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader):
    model.train()
    t = time.time()
    total_loss = 0
    for step, dic in enumerate(trainloader):
        loss, outputs = HuBMAPLoss(dic['image'], dic['mask'], model, device)
        loss.backward()
        if ((step+1)%4==0 or (step+1)==len(trainloader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss = loss.detach().item()
        total_loss += loss
        if ((step+1)%10==0 or (step+1)==len(trainloader)):
            print(
                    f'epoch {epoch} train step {step+1}/{len(trainloader)}, ' + \
                    f'loss: {total_loss/len(trainloader):.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(trainloader) else '\n', flush = True
                )

            
        
def valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader):
    model.eval()
    t = time.time()
    total_loss = 0
    for step, dic in enumerate(validloader):
        loss, outputs = HuBMAPLoss(dic['image'], dic['mask'], model, device)
        loss = loss.detach().item()
        total_loss += loss
        if ((step+1)%4==0 or (step+1)==len(validloader)):
            scheduler.step(total_loss/len(validloader))
        if ((step+1)%10==0 or (step+1)==len(validloader)):
            print(
                    f'epoch {epoch} trainz step {step+1}/{len(validloader)}, ' + \
                    f'loss: {total_loss/len(validloader):.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(validloader) else '\n' , flush = True
                )


# In[ ]:


dir_df = pd.read_csv(os.path.join(DATA_PATH, 'names.csv'))
dir_df['Folds'] = 0


# In[ ]:


FOLDS = 5
gkf = GroupKFold(FOLDS)
dir_df['Folds'] = 0
for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
    dir_df.loc[val_idx, 'Folds'] = fold


# In[ ]:



for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
    if fold>0:
        break
    
    trainloader, validloader = prepare_train_valid_dataloader(dir_df, [fold])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = HuBMAP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1)
    num_epochs = 200
    #num_epochs = 1
    for epoch in range(num_epochs):
    
        train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader)
        
        with torch.no_grad():
            valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader)
    torch.save(model.state_dict(),os.path.join('./weights',f'DICE_BCE_2_class-model.pth'))
    break




