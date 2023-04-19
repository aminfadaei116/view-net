import numpy as np
import torchvision
from configs.config import Config
from util.utils import *
from util.demo import use_webcam
from options.face2face_train_options import Face2FaceTrainOption
from models import create_model
from data import create_dataset
from util.visualizer import Visualizer
import time
from torchvision.utils import save_image


def main():
    """
    example for the command:
    --used_device YorkU --model pix2pix --name first_try --dataroot .\datasets\face2face --domain_A person_08 --domain_B person_09
    """
    opt = Face2FaceTrainOption().parse()
    config = Config(opt)


    # img_ref = torchvision.io.read_image(config.PathImg1)
    # # imgTar = torchvision.io.read_image(config.PathImg2)
    #
    # ref_key = torch.tensor(np.load(config.PathNPY1), device=config.device)
    # # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.device)
    #
    # height, width = img_ref.shape[1], img_ref.shape[2]
    # print("We are using the:", config.device)
    #
    # use_webcam(config, height, width, ref_key, img_ref)

    ##### not do this
    
    model = create_model(opt, config)  # create a model given opt.model and other options
    model.setup(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        #
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        #     if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
        #         save_result = total_iters % opt.update_html_freq == 0
        #         model.compute_visuals()
        #         visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
        #
        #     if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
        #         losses = model.get_current_losses()
        #         t_comp = (time.time() - iter_start_time) / opt.batch_size
        #         visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
        #         if opt.display_id > 0:
        #             visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
        #
        #     if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
        #         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #         save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        #         model.save_networks(save_suffix)
        #
        #     iter_data_time = time.time()
        # if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)
        #
        # print('End of epoch %d / %d \t Time Taken: %d sec' % (
        # epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # img_ref = torchvision.io.read_image(config.PathImg1)
    # # imgTar = torchvision.io.read_image(config.PathImg2)
    #
    # ref_key = torch.tensor(np.load(config.PathNPY1), device=config.device)
    # # tarKey = torch.tensor(np.load(config.PathNPY2), device=config.device)
    #
    # height, width = img_ref.shape[1], img_ref.shape[2]
    # print("We are using the:", config.device)
    #
    # use_webcam(config, height, width, ref_key, img_ref)


if __name__ == "__main__":
    main()
