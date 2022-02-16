import torch
import numpy as np
import time
from src.sap.utils import point_rasterize, grid_interp, mc_from_psr, \
calc_inters_points
from src.sap.dpsr import DPSR
import torch.nn as nn
from src.sap.network import encoder_dict, decoder_dict
from src.sap.network.utils import map2local

class PSR2Mesh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psr_grid):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)
        normals = normals.unsqueeze(0)

        res = torch.tensor(psr_grid.detach().shape[2])
        ctx.save_for_backward(verts, normals, res)

        return verts, faces, normals

    @staticmethod
    def backward(ctx, dL_dVertex, dL_dFace, dL_dNormals):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        vert_pts, normals, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())
        # matrix multiplication between dL/dV and dV/dPSR
        # dV/dPSR = - normals
        grad_vert = torch.matmul(dL_dVertex.permute(1, 0, 2), -normals.permute(1, 2, 0))
        grad_grid = point_rasterize(vert_pts, grad_vert.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid

class PSR2SurfacePoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psr_grid, poses, img_size, uv, psr_grad, mask_sample):
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts = verts * 2. - 1. # within the range of [-1, 1]

        
        p_all, n_all, mask_all = [], [], []

        for i in range(len(poses)):
            pose = poses[i]
            if mask_sample is not None:
                p_inters, mask, _, _ = calc_inters_points(verts, faces, pose, img_size, mask_gt=mask_sample[i])
            else:
                p_inters, mask, _, _ = calc_inters_points(verts, faces, pose, img_size)

            n_inters = grid_interp(psr_grad[None], (p_inters[None].detach() + 1) / 2).squeeze()
            p_all.append(p_inters)
            n_all.append(n_inters)
            mask_all.append(mask)
        p_inters_all = torch.cat(p_all, dim=0)
        n_inters_all = torch.cat(n_all, dim=0)
        mask_visible = torch.stack(mask_all, dim=0)


        res = torch.tensor(psr_grid.detach().shape[2])
        ctx.save_for_backward(p_inters_all, n_inters_all, res)

        return p_inters_all, mask_visible

    @staticmethod
    def backward(ctx, dL_dp, dL_dmask):
        pts, pts_n, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())

        # grad from the p_inters via MLP renderer
        grad_pts = torch.matmul(dL_dp[:, None], -pts_n[..., None])
        grad_grid_pts = point_rasterize((pts[None]+1)/2, grad_pts.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid_pts, None, None, None, None, None