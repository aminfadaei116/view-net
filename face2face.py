import cv2
import numpy as np
import mediapipe as mp
from numpy import save
import math as m
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from numpy import moveaxis
from numpy import asarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.linalg import multi_dot
import config
from utils import *
from matplotlib import cm
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from image2image import RenderImage
from test import UseWebcam
from generator_model import GeneratorUNet
from discriminator_model import Discriminator
from dataset import ImageDataset


def main():
    imgRef = torchvision.io.read_image(config.PathImg1)
    imgTar = torchvision.io.read_image(config.PathImg2)

    refKey = torch.tensor(np.load(config.PathNPY1), device=config.DEVICE)
    tarKey = torch.tensor(np.load(config.PathNPY2), device=config.DEVICE)

    height, width = imgRef.shape[1], imgRef.shape[2]
    print(config.DEVICE)
    
    UseWebcam(height, width, refKey, imgRef)


if __name__ == "__main__":
    main()