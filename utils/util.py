# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:28 2022

This script contains the functions of our model

@author: Amin Fadaeinejad
"""
import numpy as np
from numpy import save
import math as m
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from configs.config import *
import cv2


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        

def createMask(keys, height, width, img=None):
    
    if img != None:
      newImg = img.clone().detach().to(DEVICE)
      newImg[:, (height*keys[:, 1]).long(), (width*keys[:, 0]).long()] = 255.0
      return newImg
    else: 
      mask = torch.zeros((height, width), device=DEVICE)
      mask[(height*keys[:, 1]).long(), (width*keys[:, 0]).long()] = 1
      return mask


def createMask2(keys, height, width, img=None):
    
    if img != None:
      newImg = img.clone().detach().to(DEVICE)
      for i in range(-1, 2):
        for j in range(-1, 2):
          newImg[:, (height*keys[:, 1]).long()+i, (width*keys[:, 0]).long()+j] = 255.0
      return newImg
    else: 
      mask = torch.zeros((height, width), device=DEVICE)
      for i in range(-1, 2):
        for j in range(-1, 2):
          mask[(height*keys[:, 1]).long()+i, (width*keys[:, 0]).long()+j] = 1
      return mask

def showImageTensor(img, is3chan=True, isOutput=False, returnOutput=False):
  newImg = img.clone().detach()
  if isOutput:
    newImg = torch.squeeze(newImg)

  newImg = newImg.to('cpu') 
  if is3chan:
    # plt.imshow((newImg.permute(1, 2, 0)*255).int())
    plt.imshow((newImg.permute(1, 2, 0)))
    plt.xlabel("x axis")
    plt.ylabel("y axis")
  else:
    plt.imshow(newImg)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
  if returnOutput:
    return newImg


def draw(width, height, refKey, tarKey, size=10, connect=False):

  key1 = torch.zeros((len(refKey), 2))
  key2 = torch.zeros((len(tarKey), 2))
  key1[:, 0] = refKey[:, 0]
  key1[:, 1] = tarKey[:, 0]

  key2[:, 0] = refKey[:, 1]
  key2[:, 1] = tarKey[:, 1]


  if connect:
    for i in range(len(key1)):
        plt.plot(key1[i].cpu().numpy()*width, key2[i].cpu().numpy()*height)
      
  plt.scatter(refKey[:,0].cpu().numpy()*width, refKey[:,1].cpu().numpy()*height, s=size)
  plt.scatter(tarKey[:,0].cpu().numpy()*width, tarKey[:,1].cpu().numpy()*height, s=size)

  plt.xlim(0, width-1)
  plt.ylim(0, height-1)
  plt.xlabel("x axis")
  plt.ylabel("y axis")

  ax = plt.gca()
  ax.set_aspect('equal', adjustable='box')
  plt.gca().invert_yaxis()
  plt.show()


def TransformKeys(keys, euler, T):
    R = torch.linalg.multi_dot((Rx(euler[0]), Ry(euler[1]), Rz(euler[2])))
    center = torch.mean(keys, dim=0)
    keys = keys - center
    out = torch.matmul(keys, R)
    return out + center + T
    
def Rx(theta):
    return torch.tensor([[ 1, 0           , 0           ],
                   [ 0, torch.cos(theta),-torch.sin(theta)],
                   [ 0, torch.sin(theta), torch.cos(theta)]], device=DEVICE, dtype=torch.double)
  
def Ry(theta):
    return torch.tensor([[ torch.cos(theta), 0, torch.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-torch.sin(theta), 0, torch.cos(theta)]], device=DEVICE, dtype=torch.double)
  
def Rz(theta):
    return torch.tensor([[ torch.cos(theta), -torch.sin(theta), 0 ],
                   [ torch.sin(theta), torch.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]], device=DEVICE, dtype=torch.double)


class Feature2Feature(nn.Module):
    def __init__(self, first_layer, last_layer):
        super(Feature2Feature, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=first_layer, out_channels=8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=last_layer,kernel_size=3, padding=1),
        )
        self.id = nn.Identity()

    def forward(self, x):
        return self.layers(x)

class ViewNet(nn.Module):
    def __init__(self):
        super(ViewNet, self).__init__()
        self.Image2Feature = Feature2Feature(3, 8)
        self.Feature2Image = Feature2Feature(8, 3)

    def forward(self, height, width, refKey, tarKey, x, sd=0.01):
        Features = self.Image2Feature(x)
        TransformedFearures = RenderImage(height, width, refKey, tarKey, Features, sd=0.01, numChannel=8)
        Target = self.Feature2Image(TransformedFearures)
        return Target
        
