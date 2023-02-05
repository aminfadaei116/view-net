# -*- coding: utf-8 -*-
"""

This script contains the image renderer model

@author: Amin Fadaeinejad
"""

import torch


def interpolate_mesh(config, height, width, ref_key, tar_key, sd=0.01, eps=1e-10, dist_method="gaussian"):
    """

    :param:
        config: class Config
            A class that has the configuration parameters
        height: int
            The frames height
        width: int
            The frames width
        ref_key: torch.tensor
            Location of reference image's keypoints
        tar_key: torch.tensor
            Location of target image's keypoints
        sd: float
            Standard deviation
        eps: float
            A small value for preventing 0/0 from happening (only used in the "l2" method)
        dist_method: str ["gaussian", "l2"]
            Method of calculating the coefficient
    :return:
        mesh_flow_x: torch.tensor
        mesh_flow_y: torch.tensor
    """
    d_key = tar_key - ref_key
    flow_x, flow_y = d_key[:, 0], d_key[:, 1]
    d1 = torch.linspace(0, 1, height, device=config.device)
    d2 = torch.linspace(0, 1, width, device=config.device)
  
    meshy, meshx = torch.meshgrid(d1, d2, indexing='ij')
    del d1
    del d2
    # In order to prevent wrong behaviour we clone the mesh
    # reference https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    mx = meshx.clone()
    my = meshy.clone()
    del meshx
    del meshy

    mesh_xe = mx.expand(int(len(ref_key) / config.landmark_length), config.landmark_length, height, width)
    mesh_ye = my.expand(int(len(ref_key) / config.landmark_length), config.landmark_length, height, width)
    del mx
    del my
    
    mesh_xe = mesh_xe - tar_key[:, 0].view(-1, config.landmark_length, 1, 1)
    mesh_ye = mesh_ye - tar_key[:, 1].view(-1, config.landmark_length, 1, 1)

    if dist_method == "gaussian":
        # index 0 is for returning the max value (no need for indices)
        c = torch.max(-(mesh_xe * mesh_xe + mesh_ye * mesh_ye) / (2 * sd * sd), 1)[0]
        mesh_e = torch.exp(-(mesh_xe * mesh_xe + mesh_ye * mesh_ye) / (2 * sd * sd) - c)
    elif dist_method == "l2":
        mesh_e = 1 / (mesh_xe * mesh_xe + mesh_ye * mesh_ye + eps)
    else:
        print("Distance method not found")

    weight_meshx = mesh_e * flow_x.view(-1, config.landmark_length, 1, 1)
    weight_meshy = mesh_e * flow_y.view(-1, config.landmark_length, 1, 1)

    mesh_flow_x = 2 * torch.sum(weight_meshx, dim=1) / torch.sum(mesh_e, dim=1)
    mesh_flow_y = 2 * torch.sum(weight_meshy, dim=1) / torch.sum(mesh_e, dim=1)

    mesh_flow_x = torch.nan_to_num(mesh_flow_x)
    mesh_flow_y = torch.nan_to_num(mesh_flow_y)

    return mesh_flow_x, mesh_flow_y


def render_image(config, height, width, ref_key, tar_key, img, sd=0.01, eps=1e-10, dist_method="gaussian", num_channel=3
                 ) -> torch.tensor:
    """
    Warps the source image based on its keypoints and the destination keypoints
    :param:
        config: class Config
        height: int
            The frames height
        width: int
            The frames width
        ref_key: torch.tensor
            Location of reference image's keypoints
        tar_key: torch.tensor
            Location of target image's keypoints
        img: torch.tensor
            Source image that needs to be rendered
        sd: float
            Standard deviation
        eps: float
            A small value for preventing 0/0 from happening (only used in the "l2" method)
    :return:
        output: torch.tensor
            The rendered image using the source/target keypoints and the source image.
    """
    with torch.no_grad():
        X, Y = interpolate_mesh(config, height, width, ref_key, tar_key, sd, eps, dist_method)
        d1 = torch.linspace(-1, 1, height, device=config.device)
        d2 = torch.linspace(-1, 1, width, device=config.device)
        my, mx = torch.meshgrid(d1, d2, indexing='ij')

        meshx = mx.expand(int(len(ref_key) / config.landmark_length), height, width)
        meshy = my.expand(int(len(ref_key) / config.landmark_length), height, width)
        del mx
        del my

        meshx = meshx - X
        meshy = meshy - Y

        grid = torch.stack((meshx, meshy), 3)

        img = img.float().to(config.device)
        img = torch.reshape(img, (-1, num_channel, height, width))

        grid = grid.float()
    output = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=True)
    return output
