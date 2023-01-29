import numpy as np
import torch
import torchvision
from configs.config import Config
from utils.util import *
from utils.test import UseWebcam
# from models.generator_model import GeneratorUNet
# from models.discriminator_model import Discriminator
# from data.face2face_dataset import ImageDataset

# import os
# from pix2pix.options.test_options import TestOptions
# from pix2pix.data import create_dataset
# from pix2pix.models import create_model
# from pix2pix.util.visualizer import save_images
# from pix2pix.util import html

def main():

    config = Config("YorkU")
    imgRef = torchvision.io.read_image(config.PathImg1)
    # imgTar = torchvision.io.read_image(config.PathImg2)

    refKey = torch.tensor(np.load(config.PathNPY1), device=config.DEVICE)
    # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.DEVICE)

    height, width = imgRef.shape[1], imgRef.shape[2]
    print("We are using the: ", config.DEVICE)

    UseWebcam(config, height, width, refKey, imgRef)


if __name__ == "__main__":
    main()