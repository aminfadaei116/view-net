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
        InterpolatedFlowX: torch.tensor
        InterpolatedFlowY: torch.tensor
    """

    d_key = tar_key - ref_key
    flow_x, flow_y = d_key[:, 0], d_key[:, 1]
    d1 = torch.linspace(0, 1, height, device=config.DEVICE)
    d2 = torch.linspace(0, 1, width, device=config.DEVICE)
  
    meshy, meshx = torch.meshgrid(d1, d2, indexing='ij')
    del d1
    del d2
    # In order to prevent wrong behaviour we clone the mesh
    # reference https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    mx = meshx.clone()
    my = meshy.clone()
    del meshx
    del meshy

    MeshXE = mx.expand(int(len(ref_key) / config.landmark_length), config.landmark_length, height, width)
    MeshYE = my.expand(int(len(ref_key) / config.landmark_length), config.landmark_length, height, width)
    del mx
    del my
    
    MeshXE = MeshXE - tar_key[:, 0].view(-1, config.landmark_length, 1, 1)
    MeshYE = MeshYE - tar_key[:, 1].view(-1, config.landmark_length, 1, 1)

    if dist_method == "gaussian":
        # index 0 is for returning the max value (no need for indices)
        C = torch.max(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * sd * sd), 1)[0]       
        MeshE = torch.exp(-(MeshXE * MeshXE + MeshYE * MeshYE) / (2 * sd * sd) - C)
    elif dist_method == "l2":
        MeshE = 1/(MeshXE * MeshXE + MeshYE * MeshYE + eps)
    else:
        print("Distance method not found")

    WeightMeshX = MeshE * flow_x.view(-1, config.landmark_length, 1, 1)
    WeightMeshY = MeshE * flow_y.view(-1, config.landmark_length, 1, 1)

    InterpolatedFlowX = 2 * torch.sum(WeightMeshX, dim=1)/torch.sum(MeshE, dim=1)
    InterpolatedFlowY = 2 * torch.sum(WeightMeshY, dim=1)/torch.sum(MeshE, dim=1)

    InterpolatedFlowX = torch.nan_to_num(InterpolatedFlowX)
    InterpolatedFlowY = torch.nan_to_num(InterpolatedFlowY)

    return InterpolatedFlowX, InterpolatedFlowY


def render_image(config, height, width, ref_key, tar_key, img, sd=0.01, eps=1e-10, dist_method="gaussian", num_channel=3
                 ) -> torch.tensor:
    """
    Warps the source image based on its keypoints and the destination keypoints
    :param:
        config: class Config
        height: int
        width: int
        ref_key: torch.tensor
        tar_key: torch.tensor
        img: torch.tensor
        sd: float
        eps:
    :return:
        output: torch.tensor
            The rendered image using the source/target keypoints and the source image.
    """
    with torch.no_grad():
        X, Y = interpolate_mesh(config, height, width, ref_key, tar_key, sd, eps, dist_method)
        d1 = torch.linspace(-1, 1, height, device=config.DEVICE)
        d2 = torch.linspace(-1, 1, width, device=config.DEVICE)
        my, mx = torch.meshgrid(d1, d2, indexing='ij')

        meshx = mx.expand(int(len(ref_key) / config.landmark_length), height, width)
        meshy = my.expand(int(len(ref_key) / config.landmark_length), height, width)
        del mx
        del my

        meshx = meshx - X
        meshy = meshy - Y

        grid = torch.stack((meshx, meshy), 3)

        img = img.float().to(config.DEVICE)
        img = torch.reshape(img, (-1, num_channel, height, width))

        grid = grid.float()
    output = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=True)
    return output
