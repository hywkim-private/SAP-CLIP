import torch
import clip
from src.sap.encode2points import Encode2Points
from src.generate import generate_encoder
from src.sap.network.encoder import LocalPoolPointnet
from src.sap.network.decoder import LocalDecoder, OccupancyNet
#A network that takes pointcloud inputs and outputs offset parameters
#We will take a prep-trained Conv Occupancy network and reset the parameters of th decoder
class SAP_CLIP:
    def __init__(self, cfg, clip_model, device):
        super().__init__()
        self.cfg = cfg
        dim = cfg['data']['dim'] 
        c_dim = cfg['model']['c_dim']
        decoder_kwargs = cfg['model']['decoder_kwargs']
        out_dim_offset = 3
        #this part is hard-coded for convenience=>add this to the config
        self.device = device
        self.clip_model = clip_model
        self.pointcloud = None
        #TEMPORARY PARAM=>SHOULD BE INCLUDED IN CONFIG
        #HARD-CODED FOR TESTING PURPOSE
        #define the dim of text feature, output from clip
        self.t_dim = 512
        self.p_dim = 32
        self.num_pt = 2
        self.n_blocks = 5 
        self.hidden_size = 512
        #generate a pretrained encodetopoinnts_model
        #only take the encoder layer since we will build a custom decoder 
        self.encoder = generate_encoder(cfg, self.device)
        #a decoder that yields occupancy probabilities
        self.decoder_occupancy = OccupancyNet(t_dim=self.t_dim, p_dim=self.p_dim, 
        num_pt=self.num_pt, n_blocks=self.n_blocks, hidden_size=self.hidden_size, device=self.device)
        
    #run the input text prompt on the specified clip model and get the text feature map
    def get_text_feat(self, t):
        t = clip.tokenize(t).to(self.device)
        #get the text feature map
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(t)
        return text_feat
          
    #given list of point coordinates to sample, return the feature vector for each points
    def get_point_feat(self, p):
        pt_feat = self.encoder.forward(p)
        return pt_feat
        
        
    #given text and points, and a point coordinate to sample, yield the occupancy probs
    def forward(self, t, p, sample_grid):
        pt_feat = self.get_point_feat(p)
        text_feat = self.get_text_feat(t)
        sampled_feat = self.decoder_occupancy.sample_features(sample_grid, pt_feat)
        occupancy_prob = self.decoder_occupancy.forward(sampled_feat, text_feat)
        return occupancy_prob
    
        