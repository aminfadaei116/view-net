# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:28 2022

This script contains the functions of our model

@author: afadaei
"""

from numpy import save
import math as m
import matplotlib.pyplot as plt
from HyperParameters import *



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


# def measureWeight(x, y, data2, dKey, InterCov, distMethod):
#     Sum = torch.zeros(1, device=DEVICE)
#     Wsum = torch.zeros(1, device=DEVICE)
    
#     if(distMethod == "gaussian"):
#         d1 = torch.linspace(0, 1, height, device=DEVICE)
#         d2 = torch.linspace(0, 1, width, device=DEVICE)

#         meshx, meshy = torch.meshgrid(d1, d2, indexing='ij') 
#         mx = meshx.clone()
#         my = meshy.clone()
#         MeshXE = mx.expand(478, 480, 640)
#         MeshYE = my.expand(478, 480, 640)


#         MeshXE = MeshXE - refKey[:, 1].view(-1, 1, 1)
#         MeshYE = MeshYE - refKey[:, 0].view(-1, 1, 1)

#         MeshE = torch.exp(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * 0.01 * 0.01))
        
#         WeightMeshX = MeshE * flowX.view(-1, 1, 1)
#         WeightMeshY = MeshE * flowX.view(-1, 1, 1)

#         InterpolatedFlowX = torch.sum(WeightMeshX, dim=0)/torch.sum(MeshE, dim=0)
#         InterpolatedFlowY = torch.sum(WeightMeshY, dim=0)/torch.sum(MeshE, dim=0)
#         return InterpolatedFlowX, InterpolatedFlowY
#     if(distMethod == "l2"):
#         for i in range(len(data2)):
#             point = [data2[i, 1]-x, data2[i, 0]-y]
#             w = 1/(point[0]*point[0] + point[1]*point[1]+0.0001)
#             Wsum += w*dKey[i]
#             Sum += w
#         return Wsum/Sum
#     elif(distMethod == "nn"):
#         d = torch.zeros((len(data2)), dtype=torch.float64)
#         numClosest = 4
#         for i in range(len(data2)):
#             d[i] = (data2[i, 1]-x)**2 + (data2[i, 0]-y)**2 
#         idx = np.argpartition(d, numClosest)
#         kk = idx[:numClosest]
#         return sum(dKey[kk])/numClosest
    

def MyInterpol(height, width, dataRef, dataTarg, sd=0.01, eps=1e-10, distMethod="gaussian"):
    
  
    dKey = dataTarg - dataRef 


    flowX, flowY = dKey[:,0], dKey[:,1]
    d1 = torch.linspace(0, 1, height, device=DEVICE)
    d2 = torch.linspace(0, 1, width, device=DEVICE)

    # meshx, meshy = torch.meshgrid(d1, d2, indexing='ij') 
    meshy, meshx = torch.meshgrid(d1, d2, indexing='ij')
    # mx = meshx.clone()
    # my = meshy.clone()
    MeshXE = meshx.expand(len(dataRef), 480, 640)
    MeshYE = meshy.expand(len(dataRef), 480, 640)


    MeshXE = MeshXE - dataTarg[:, 0].view(-1, 1, 1)
    MeshYE = MeshYE - dataTarg[:, 1].view(-1, 1, 1)
    if distMethod == "gaussian":
      MeshE = torch.exp(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * sd * sd))
    elif distMethod == "l2":
      MeshE = 1/(MeshXE * MeshXE + MeshYE * MeshYE + eps)
    else:
      print("Distance method not found")
    WeightMeshX = MeshE * flowX.view(-1, 1, 1)
    WeightMeshY = MeshE * flowY.view(-1, 1, 1)

    InterpolatedFlowX = torch.sum(WeightMeshX, dim=0)/torch.sum(MeshE, dim=0)
    InterpolatedFlowY = torch.sum(WeightMeshY, dim=0)/torch.sum(MeshE, dim=0)

    return 2 * InterpolatedFlowX, 2 * InterpolatedFlowY


def RenderImage(height, width, refKey, tarKey, img, sd=0.01, eps=1e-10, distMethod="gaussian"):
    X, Y = MyInterpol(height, width, refKey, tarKey, sd, eps, distMethod)
    d1 = torch.linspace(-1, 1, height)
    d2 = torch.linspace(-1, 1, width)
    meshy, meshx = torch.meshgrid(d1, d2, indexing='ij')
    # meshx, meshy = meshx.to(DEVICE), meshy.to(DEVICE)

    meshx = meshx.clone().detach().to(DEVICE)
    meshy = meshy.clone().detach().to(DEVICE)

    meshx = meshx - X
    meshy = meshy - Y

    grid = torch.stack((meshx, meshy), 2)
    grid = grid.unsqueeze(0)
    #img = torch.tensor(img, dtype=torch.float, device=DEVICE)
    img = img.float().to(DEVICE)
    img = torch.unsqueeze(img, 0)
    # grid = torch.tensor(grid, dtype=torch.float)
    grid = grid.float()
    output = torch.nn.functional.grid_sample(img, grid, padding_mode="border",align_corners=True)
    return output


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