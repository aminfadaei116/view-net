import numpy as np
import torchvision
from configs.config import Config
from utils.util import *
from utils.test import use_webcam


def main():

    config = Config("YorkU")
    img_ref = torchvision.io.read_image(config.PathImg1)
    # imgTar = torchvision.io.read_image(config.PathImg2)

    ref_key = torch.tensor(np.load(config.PathNPY1), device=config.DEVICE)
    # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.DEVICE)

    height, width = img_ref.shape[1], img_ref.shape[2]
    print("We are using the: ", config.DEVICE)

    use_webcam(config, height, width, ref_key, img_ref)


if __name__ == "__main__":
    main()
