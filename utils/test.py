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
import configs.config as config
import cv2
from models.image2image import RenderImage
import torch


def UseWebcam(height, width, refKey, imgRef):
  # For webcam input:
  cap = cv2.VideoCapture(0)
  with config.mp_face_mesh.FaceMesh(
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
        LandMarks = np.zeros((config.FACE_LANKMARK_LENGTH, 3))
        for i, land in enumerate(results.multi_face_landmarks[0].landmark):
    
          LandMarks[i] = [land.x, land.y, land.z]
        
        landTen = torch.tensor(LandMarks, device=config.DEVICE, requires_grad = False)
        ##
        with torch.no_grad():
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