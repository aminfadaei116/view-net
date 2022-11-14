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


def showImageTensor(img, is3chan=True, isOutput=False, returnOutput=False):
  newImg = img.clone().detach()
  if isOutput:
    newImg = torch.squeeze(newImg)
    
  newImg = newImg.to('cpu')
  
  if is3chan:
    # plt.imshow((newImg.permute(1, 2, 0)*255).int())
    plt.imshow((newImg.permute(1, 2, 0)))
  else:
    plt.imshow(newImg)
  if returnOutput:
    return newImg

def createDiffMask(dataRef, dataTarg):
    dKey = dataRef - dataTarg
    return dKey[:,0], dKey[:,1]


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
    

def MyInterpol(height, width, dataRef, dataTarg, sd, distMethod):
    dKey = dataRef - dataTarg
    flowX, flowY = dKey[:,0], dKey[:,1]
    d1 = torch.linspace(0, 1, height, device=DEVICE)
    d2 = torch.linspace(0, 1, width, device=DEVICE)

    meshx, meshy = torch.meshgrid(d1, d2, indexing='ij') 
    # mx = meshx.clone()
    # my = meshy.clone()
    MeshXE = meshx.expand(478, 480, 640)
    MeshYE = meshy.expand(478, 480, 640)


    MeshXE = MeshXE - dataRef[:, 1].view(-1, 1, 1)
    MeshYE = MeshYE - dataRef[:, 0].view(-1, 1, 1)

    MeshE = torch.exp(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * sd * sd))
    WeightMeshX = MeshE * flowX.view(-1, 1, 1)
    WeightMeshY = MeshE * flowY.view(-1, 1, 1)

    InterpolatedFlowX = torch.sum(WeightMeshX, dim=0)/torch.sum(MeshE, dim=0)
    InterpolatedFlowY = torch.sum(WeightMeshY, dim=0)/torch.sum(MeshE, dim=0)
    return 2 * InterpolatedFlowX, 2 * InterpolatedFlowY


def RenderImage(height, width, refKey, tarKey, img):
    X, Y = MyInterpol(height, width, refKey, tarKey, 0.1, "nn")
    d1 = torch.linspace(-1, 1, height)
    d2 = torch.linspace(-1, 1, width)
    meshx, meshy = torch.meshgrid(d1, d2, indexing='ij')
    # meshx, meshy = meshx.to(DEVICE), meshy.to(DEVICE)

    meshx = meshx.clone().detach().to(DEVICE)
    meshy = meshy.clone().detach().to(DEVICE)

    meshx = meshx + Y
    meshy = meshy + X
    grid = torch.stack((meshy, meshx), 2)
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