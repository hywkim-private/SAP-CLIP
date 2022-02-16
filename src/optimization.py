import time, os
import numpy as np
from src.clip.render import Renderer
import clip
import torch
from torch.nn import functional as F
import trimesh
from torchvision import transforms
from pytorch3d.structures import Meshes
from src.clip.mesh import Mesh
from src.sap.dpsr import DPSR
from src.sap.psr2mesh import PSR2Mesh
from src.sap.utils import grid_interp, verts_on_largest_mesh, \
    export_pointcloud, mc_from_psr, GaussianSmoothing

from torchvision.utils import save_image
from torchvision.io import write_video
from pytorch3d.loss import chamfer_distance
import open3d as o3d
'''
render = Renderer()

device = "cuda:0"
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
prompt_token = clip.tokenize(["a 3d rendering of a dog"]).to(device)
encoded_text = clip_model.encode_text(prompt_token)

clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

])'''
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# CLIP Transform
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    clip_normalizer
])
render = Renderer()

class Trainer(object):
    '''
    Args:
        cfg       : config file
        optimizer : pytorch optimizer object
        clip_model: clip_model 
        device    : pytorch device
    '''

    def __init__(self, cfg, optimizer, clip_model, device=None):
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.psr2mesh = PSR2Mesh.apply
        self.data_type = cfg['data']['data_type']
        self.clip_model = clip_model
        # initialize DPSR
        self.dpsr = DPSR(res=(cfg['model']['grid_res'],
                              cfg['model']['grid_res'],
                              cfg['model']['grid_res']),
                         sig=cfg['model']['psr_sigma'])
        if torch.cuda.device_count() > 1:
            self.dpsr = torch.nn.DataParallel(self.dpsr)  # parallell DPSR
        self.dpsr = self.dpsr.to(device)

    def train_step(self, pt, text, it):
        ''' Performs a training step.
        Args:
            pt (torch.tensor)    : input point clouds
            text (str)  : input text string
            model (nn.Module or None): a neural network or None
            it (int)                 : the number of iterations
        '''

        self.optimizer.zero_grad()
        loss, loss_each = self.compute_loss(pt,text, it)

        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_each

    def compute_loss(self, pt, text, it=0):
        '''  Compute the loss.
        Args:
            pt                       : pointcloud input
            text (str)               : text input
            inputs (torch.tensor)    : input point clouds
            it (int)                 : the number of iterations
        '''

        device = self.device
        res = self.cfg['model']['grid_res']

        # source oriented point clouds to PSR grid
        psr_grid, points, normals = self.pcl2psr(pt)

        # build mesh
        v, f, n = self.psr2mesh(psr_grid)

        # the output is in the range of [0, 1), we make it to the real range [0, 1]. 
        # This is a hack for our DPSR solver
        v = v * res / (res - 1)

        points = points * 2. - 1.
        v = v * 2. - 1.  # within the range of (-1, 1)
        
        #HARD-CODDED TO USE DIFFERENTIABLE RENDER FUNCTIONS
        mesh = Meshes(verts=v, faces=f)
        #set faces to int type since we'll be using faces as index to camera-projected vertices
        f = f.squeeze(0).type(torch.int64)
        v = v.squeeze(0)
        v_normal = mesh.verts_normals_packed()
        f_normal = mesh.faces_normals_packed()
        #use the Mesh class defined in src.clip.mesh
        sampled_mesh = Mesh(v, f, v_normal, f_normal)
        loss = 0
        loss_each = {}
        rendered_images, elev, azim = render.render_front_views(sampled_mesh, num_views=4,
                                                                show=False,
                                                                center_azim=0,
                                                                center_elev=0,
                                                                std=8,
                                                                return_views=True,
                                                                background=torch.tensor([1., 1., 1.], device = device))
        clip_image = clip_transform(rendered_images)
        encoded_renders = self.clip_model.encode_image(clip_image)
        tokenized_text = clip.tokenize(text).to(self.device)
        encoded_text = self.clip_model.encode_text(tokenized_text)
        loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
        return loss, loss_each

    def pcl2psr(self, inputs):
        '''  Convert an oriented point cloud to PSR indicator grid
        Args:
            inputs (torch.tensor): input oriented point clouds
        '''

        points, normals = inputs[..., :3], inputs[..., 3:]
        if self.cfg['model']['apply_sigmoid']:
            points = torch.sigmoid(points)
        if self.cfg['model']['normal_normalize']:
            normals = normals / normals.norm(dim=-1, keepdim=True)

        # DPSR to get grid
        psr_grid = self.dpsr(points, normals).unsqueeze(1)
        psr_grid = torch.tanh(psr_grid)

        return psr_grid, points, normals

    

    def point_resampling(self, inputs):
        '''  Resample points
        Args:
            inputs (torch.tensor): oriented point clouds
        '''

        psr_grid, points, normals = self.pcl2psr(inputs)

        # shortcuts
        n_grow = self.cfg['train']['n_grow_points']

        # [hack] for points resampled from the mesh from marching cubes, 
        # we need to divide by s instead of (s-1), and the scale is correct.
        verts, faces, _ = mc_from_psr(psr_grid, real_scale=False, zero_level=0)

        # find the largest component
        pts_mesh, faces_mesh = verts_on_largest_mesh(verts, faces)

        # sample vertices only from the largest component, not from fragments
        mesh = trimesh.Trimesh(vertices=pts_mesh, faces=faces_mesh)
        pi, face_idx = mesh.sample(n_grow + points.shape[1], return_index=True)
        normals_i = mesh.face_normals[face_idx].astype('float32')
        pts_mesh = torch.tensor(pi.astype('float32')).to(self.device)[None]
        n_mesh = torch.tensor(normals_i).to(self.device)[None]

        points, normals = pts_mesh, n_mesh
        print('{} total points are resampled'.format(points.shape[1]))

        # update inputs
        points = torch.log(points / (1 - points))  # inverse sigmoid
        inputs = torch.cat([points, normals], dim=-1)
        inputs.requires_grad = True

        return inputs

    def visualize(self, data, inputs, renderer, epoch, o3d_vis=None):
        '''  Visualization.
        Args:
            data (dict)                 : data dictionary
            inputs (torch.tensor)       : source point clouds
            renderer (nn.Module or None): a neural network or None
            epoch (int)                 : the number of iterations
            o3d_vis (o3d.Visualizer)    : open3d visualizer
        '''

        data_type = self.cfg['data']['data_type']
        it = '{:04d}'.format(int(epoch / self.cfg['train']['visualize_every']))

        if (self.cfg['train']['exp_mesh']) \
                | (self.cfg['train']['exp_pcl']) \
                | (self.cfg['train']['o3d_show']):
            psr_grid, points, normals = self.pcl2psr(inputs)

            with torch.no_grad():
                v, f, n = mc_from_psr(psr_grid, pytorchify=True,
                                      zero_level=self.cfg['data']['zero_level'], real_scale=True)
                v, f, n = v[None], f[None], n[None]

                v = v * 2. - 1.  # change to the range of [-1, 1]

            color_v = None
            if data_type == 'img':
                if self.cfg['train']['vis_vert_color'] & \
                        (self.cfg['train']['l_weight']['rgb'] != 0.):
                    color_v = renderer['color'](v, n).squeeze().detach().cpu().numpy()
                    color_v[color_v < 0], color_v[color_v > 1] = 0., 1.

            vv = v.detach().squeeze().cpu().numpy()
            ff = f.detach().squeeze().cpu().numpy()
            points = points * 2 - 1
            visualize_points_mesh(o3d_vis, points, normals,
                                  vv, ff, self.cfg, it, epoch, color_v=color_v)

        else:
            v, f, n = inputs

        if (data_type == 'img') & (self.cfg['train']['vis_rendering']):
            pred_imgs = []
            pred_masks = []
            n_views = len(data['poses'])
            # idx_list = trange(n_views)
            idx_list = [13, 24, 27, 48]

            # !
            model = renderer.eval()
            for idx in idx_list:
                pose = data['poses'][idx]
                rgb = data['rgbs'][idx]
                mask_gt = data['masks'][idx]
                img_size = rgb.shape[0] if rgb.shape[0] == rgb.shape[1] else (rgb.shape[0], rgb.shape[1])
                ray = None
                if 'rays' in data.keys():
                    ray = data['rays'][idx]
                if self.cfg['train']['l_weight']['rgb'] != 0.:
                    fea_grid = None
                    if model.unet3d is not None:
                        with torch.no_grad():
                            fea_grid = model.unet3d(psr_grid).permute(0, 2, 3, 4, 1)
                    if model.encoder is not None:
                        pp = torch.cat([(points + 1) / 2, normals], dim=-1)
                        fea_grid = model.encoder(pp,
                                                 normalize=False).permute(0, 2, 3, 4, 1)

                    pred, visible_mask = render_rgb(v, f, n, pose,
                                                    model.rendering_network.eval(),
                                                    img_size, ray=ray, fea_grid=fea_grid)
                    img_pred = torch.zeros([rgb.shape[0] * rgb.shape[1], 3])
                    img_pred[visible_mask] = pred.detach().cpu()

                    img_pred = img_pred.reshape(rgb.shape[0], rgb.shape[1], 3)
                    img_pred[img_pred < 0], img_pred[img_pred > 1] = 0., 1.
                    filename = os.path.join(self.cfg['train']['dir_rendering'],
                                            'rendering_{}_{:d}.png'.format(it, idx))
                    save_image(img_pred.permute(2, 0, 1), filename)
                    pred_imgs.append(img_pred)

                # ! Mesh rendering using Phong shading model
                filename = os.path.join(self.cfg['train']['dir_rendering'],
                                        'mesh_{}_{:d}.png'.format(it, idx))
                visualize_mesh_phong(v, f, n, pose, img_size, name=filename)

            if len(pred_imgs) >= 1:
                pred_imgs = torch.stack(pred_imgs, dim=0)
                save_image(pred_imgs.permute(0, 3, 1, 2),
                           os.path.join(self.cfg['train']['dir_rendering'],
                                        '{}.png'.format(it)), nrow=4)
                if self.cfg['train']['save_video']:
                    write_video(os.path.join(self.cfg['train']['dir_rendering'],
                                             '{}.mp4'.format(it)),
                                (pred_imgs * 255.).type(torch.uint8), fps=24)

    def save_mesh_pointclouds(self, inputs, epoch, center=None, scale=None):
        '''  Save meshes and point clouds.
        Args:
            inputs (torch.tensor)       : source point clouds
            epoch (int)                 : the number of iterations
            center (numpy.array)        : center of the shape
            scale (numpy.array)         : scale of the shape
        '''

        exp_pcl = self.cfg['train']['exp_pcl']
        exp_mesh = self.cfg['train']['exp_mesh']

        psr_grid, points, normals = self.pcl2psr(inputs)

        if exp_pcl:
            dir_pcl = self.cfg['train']['dir_pcl']
            p = points.squeeze(0).detach().cpu().numpy()
            p = p * 2 - 1
            n = normals.squeeze(0).detach().cpu().numpy()
            if scale is not None:
                p *= scale
            if center is not None:
                p += center
            export_pointcloud(os.path.join(dir_pcl, '{:04d}.ply'.format(epoch)), p, n)
        if exp_mesh:
            dir_mesh = self.cfg['train']['dir_mesh']
            with torch.no_grad():
                v, f, _ = mc_from_psr(psr_grid,
                                      zero_level=self.cfg['data']['zero_level'], real_scale=True)
                v = v * 2 - 1
                if scale is not None:
                    v *= scale
                if center is not None:
                    v += center
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector(f)
            outdir_mesh = os.path.join(dir_mesh, '{:04d}.ply'.format(epoch))
            o3d.io.write_triangle_mesh(outdir_mesh, mesh)

        if self.cfg['train']['vis_psr']:
            dir_psr_vis = self.cfg['train']['out_dir'] + '/psr_vis_all'
            visualize_psr_grid(psr_grid, out_dir=dir_psr_vis)