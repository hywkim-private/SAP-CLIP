import torch
from src.sap.encode2points import Encode2Points

def generate_encoder(cfg, device):
    model = Encode2Points(cfg).to(device)
    pretrained_model = torch.load('./src/sap/encode2points.pt', map_location=torch.device(device))
    model.load_state_dict(pretrained_model["state_dict"])
    for param_tensor in model.state_dict():
        model.state_dict()[param_tensor].requires_grad = False
    encoder = model.encoder
    return encoder