# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:55:15 2022

This script contains the hyper parameters of our model

@author: afadaei
"""

import torch

PathNPY1 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_7\Image_134.npy"
PathNPY2 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_7\Image_182.npy"

PathImg1 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_7\Image_134.jpg"
PathImg2 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_7\Image_182.jpg"

DEVICE=torch.device('cuda')
pi = 3.14159265359