import numpy as np
import torchvision
from configs.config import Config
from util.util import *
from util.demo import use_webcam
from options.face2face_options import Face2FaceOption


def main():
    parser = Face2FaceOption().get_parser()
    config = Config(parser)
    img_ref = torchvision.io.read_image(config.PathImg1)
    # imgTar = torchvision.io.read_image(config.PathImg2)

    ref_key = torch.tensor(np.load(config.PathNPY1), device=config.device)
    # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.device)

    height, width = img_ref.shape[1], img_ref.shape[2]
    print("We are using the:", config.device)

    use_webcam(config, height, width, ref_key, img_ref)


if __name__ == "__main__":
    main()
