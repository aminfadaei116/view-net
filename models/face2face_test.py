import numpy as np
import torchvision
from configs.config import Config
from util.utils import *
from util.demo import use_webcam
from options.face2face_test_options import Face2FaceTestOption
from options.face2face_train_options import Face2FaceTrainOption
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer



def main():
    """
    example for the command:
    --used_device YorkU --model pix2pix --name first_try --dataroot test
    """
    parser = Face2FaceTestOption().get_parser()
    config = Config(parser)
    # model = create_model(parser)  # create a model given opt.model and other options
    # model.setup(parser)

    img_ref = torchvision.io.read_image(config.PathImg1)
    # imgTar = torchvision.io.read_image(config.PathImg2)

    ref_key = torch.tensor(np.load(config.PathNPY1), device=config.device)
    # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.device)

    height, width = img_ref.shape[1], img_ref.shape[2]
    print("We are using the:", config.device)

    use_webcam(config, height, width, ref_key, img_ref)


if __name__ == "__main__":
    main()
