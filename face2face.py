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

    config = Config("Ubisoft")
    imgRef = torchvision.io.read_image(config.PathImg1)
    # imgTar = torchvision.io.read_image(config.PathImg2)

    refKey = torch.tensor(np.load(config.PathNPY1), device=config.DEVICE)
    # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.DEVICE)

    height, width = imgRef.shape[1], imgRef.shape[2]
    print("We are using the: ", config.DEVICE)

    UseWebcam(config, height, width, refKey, imgRef)


def main2():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

if __name__ == "__main__":
    main()