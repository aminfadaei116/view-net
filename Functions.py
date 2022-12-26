# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:28 2022

This script contains the functions of our model

@author: afadaei
"""
import numpy as np
from numpy import save
import math as m
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from HyperParameters import *
import cv2



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

def MyInterpol(height, width, dataRef, dataTarg, sd=0.01, eps=1e-10, distMethod="gaussian"):
  




    dKey = dataTarg - dataRef 

    flowX, flowY = dKey[:,0], dKey[:,1]
    d1 = torch.linspace(0, 1, height, device=DEVICE)
    d2 = torch.linspace(0, 1, width, device=DEVICE)
  
    meshy, meshx = torch.meshgrid(d1, d2, indexing='ij')
    del d1
    del d2
    # In order to prevent wrong behaviour we clone the mesh
    # reference https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    mx = meshx.clone()
    my = meshy.clone()
    del meshx
    del meshy

    MeshXE = mx.expand(int(len(dataRef)/FACE_LANKMARK_LENGTH), FACE_LANKMARK_LENGTH, height, width)
    MeshYE = my.expand(int(len(dataRef)/FACE_LANKMARK_LENGTH), FACE_LANKMARK_LENGTH, height, width)
    del mx
    del my
    
    MeshXE = MeshXE - dataTarg[:, 0].view(-1, FACE_LANKMARK_LENGTH, 1, 1)
    MeshYE = MeshYE - dataTarg[:, 1].view(-1, FACE_LANKMARK_LENGTH, 1, 1)


    if distMethod == "gaussian":
        # index 0 is for returning the max value (no need for indices)
        C = torch.max(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * sd * sd), 1)[0]       
        MeshE = torch.exp(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * sd * sd) - C)
    elif distMethod == "l2":
        MeshE = 1/(MeshXE * MeshXE + MeshYE * MeshYE + eps)
    else:
        print("Distance method not found")

    WeightMeshX = MeshE * flowX.view(-1, FACE_LANKMARK_LENGTH, 1, 1)
    WeightMeshY = MeshE * flowY.view(-1, FACE_LANKMARK_LENGTH, 1, 1)

    InterpolatedFlowX = torch.sum(WeightMeshX, dim=1)/torch.sum(MeshE, dim=1)
    InterpolatedFlowY = torch.sum(WeightMeshY, dim=1)/torch.sum(MeshE, dim=1)

    InterpolatedFlowX = torch.nan_to_num(InterpolatedFlowX)
    InterpolatedFlowY = torch.nan_to_num(InterpolatedFlowY)

    return 2 * InterpolatedFlowX, 2 * InterpolatedFlowY


def RenderImage(height, width, refKey, tarKey, img, sd=0.01, eps=1e-10, distMethod="gaussian", numChannel=3):


    with torch.no_grad():
        X, Y= MyInterpol(height, width, refKey, tarKey, sd, eps, distMethod)
        d1 = torch.linspace(-1, 1, height, device=DEVICE)
        d2 = torch.linspace(-1, 1, width, device=DEVICE)
        my, mx = torch.meshgrid(d1, d2, indexing='ij')

        meshx = mx.expand(int(len(refKey)/FACE_LANKMARK_LENGTH), height, width)
        meshy = my.expand(int(len(refKey)/FACE_LANKMARK_LENGTH), height, width)
        del mx
        del my

        meshx = meshx - X
        meshy = meshy - Y

        grid = torch.stack((meshx, meshy), 3)

        img = img.float().to(DEVICE)
        img = torch.reshape(img, (-1, numChannel, height, width))

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

def UseWebcam():
  # For webcam input:
  cap = cv2.VideoCapture(0)
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        LandMarks = np.zeros((FACE_LANKMARK_LENGTH, 3))
        for i, land in enumerate(results.multi_face_landmarks[0].landmark):
    
          LandMarks[i] = [land.x, land.y, land.z]
        
        landTen = torch.tensor(LandMarks, device=DEVICE)
        ##
        output = RenderImage(height, width, refKey, landTen, imgRef, sd=0.01)

        dummy = torch.squeeze(output)
        # dummy = createMask(landTen, height, width, dummy).int()
        image =  (dummy.cpu().permute(1, 2, 0).numpy()).astype(np.uint8)
        #img = showImageTensor(dummy, is3chan=True, isOutput=True, returnOutput=True)
  # showImageTensor(output.int(), is3chan=True, isOutput=True)
  # showImageTensor(output, isOutput=True)


        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      cv2.imshow('MediaPipe Face Mesh', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break
  cap.release()
  cv2.destroyAllWindows()


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
        
