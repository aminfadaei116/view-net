import numpy as np
import torch
import torchvision
import configs.config as config
from utils.util import *
from matplotlib import cm
from torchvision import datasets
import torch.optim as optim
from models.image2image import RenderImage
from utils.test import UseWebcam
from models.generator_model import GeneratorUNet
from models.discriminator_model import Discriminator
from data.dataset import ImageDataset


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