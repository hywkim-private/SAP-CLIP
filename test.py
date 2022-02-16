from src.sap.encode2points import Encode2Points
from src.optimization import Trainer
from src.utils import update_recursive, sample_coordinates_uniform
from src.vis import visualize_pointclouds
from src.model import SAP_CLIP
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.structures import Pointclouds

import clip
import torch
import numpy as np
import yaml
from plyfile import PlyData



def main():
  
  #device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device: {device}")
  # Load configuration from file itself
  with open('./configs/learning.yaml', 'r') as f:
    cfg_learn = yaml.load(f, Loader=yaml.Loader)
  with open('./configs/default.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
  update_recursive(cfg, cfg_learn)
  model = Encode2Points(cfg).to(device)
  

  #test read lst file
  data_path = "./data/demo/wheel.ply"
  plydata = PlyData.read(data_path)
  vertices = np.stack([plydata['vertex']['x'],
                      plydata['vertex']['y'],
                      plydata['vertex']['z']], axis=1)
  normals = np.stack([plydata['vertex']['nx'],
                      plydata['vertex']['ny'],
                      plydata['vertex']['nz']], axis=1)
  N = vertices.shape[0]
  center = vertices.mean(0)
  scale = np.max(np.max(np.abs(vertices - center), axis=0))
  vertices -= center
  vertices /= scale
  vertices *= 0.9
  #prepare input tensors
  pts = torch.tensor(vertices, device=device)[None].float()
  normals = torch.tensor(normals, device=device)[None].float()
  inputs = torch.cat([pts, normals], axis=-1).float()
  inputs.requires_grad = True

  #visualize
  #test_fig = visualize_pointclouds(pts)
  #test_fig.write_image("./data/test.png", format='png')
  
  #Run occupancy probability network
  test_text = "a dog with spikes"
  clip_model_name = "ViT-B/32"
  clip_model, _ = clip.load(clip_model_name, device=device)
  #initialize the SAP_CLIP model
  sap_clip = SAP_CLIP(cfg, clip_model, device)
  #sample points for which to caluualte occupancy probs
  num_samples = 20
  sample_grid = torch.tensor(sample_coordinates_uniform(num_samples)).unsqueeze(0).to(device)
  print(sample_grid.shape)
  occupancy_prob = sap_clip.forward(test_text, pts, sample_grid)
  print(f"Occupancy probabilities for {num_samples} sample points: {occupancy_prob}")

  #test for Shape as points optimization based model
  epoch = 100
  #THIS MUST BE CHANGED
  for i in range(epoch):
    optimizer = torch.optim.Adam([inputs], lr =0.0001)
    trainer = Trainer(cfg, optimizer, clip_model,device=device)
    loss, loss_each = trainer.train_step(inputs, test_text, it=epoch)
    print(f"Optimizer loss for epoch {i}: {loss}")
    
if __name__ == '__main__':
  main()
    