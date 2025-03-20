'''
 * Adapted from ImageReward (https://github.com/THUDM/ImageReward)
'''
import os
import torch
import torch.nn as nn
from PIL import Image
from .models.BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, 1)        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        rewards = self.fc3(x)
        return rewards

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input,get_feature=None):
        if get_feature ==None:
            return self.layers(input)
        else:
            for layer in self.layers:
                input = layer(input)
                if isinstance(layer, nn.Linear) and input.size(1) == get_feature:
                    return input

class CrossViewFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_views=4):
        super().__init__()
        self.num_views = num_views
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=12,  
            dropout=0.1,
            batch_first=False
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        total_samples, dim = x.shape
        bs = total_samples // self.num_views
        seq_features = x.view(bs, self.num_views, dim).permute(1, 0, 2)
        attn_output, _ = self.self_attn(
            query=seq_features,  
            key=seq_features,
            value=seq_features
        )  
        combined = torch.cat([seq_features, attn_output], dim=-1) 
        gate = self.fusion_gate(combined)
        fused_seq = gate * attn_output + (1 - gate) * seq_features
        return fused_seq.mean(dim=0) 

class Reward3D(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.cross_view_adapter = CrossViewFusion(hidden_dim=768, num_views=4)
        self.mlp = MLP(768)
        self.mean = 0.16717362830052426
        self.std = 5
        self.freeze_except_cross_adapter()

    def freeze_except_cross_adapter(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.cross_view_adapter.parameters():
            param.requires_grad = True

    def forward(self, image,prompt_ids, prompt_attention_mask):
        image_embeds = self.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt_ids,
                                                    attention_mask = prompt_attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
        view_features = text_output.last_hidden_state[:,0,:] 
        fused_feature = self.cross_view_adapter(view_features)
        rewards = self.mlp(fused_feature)
        rewards = (rewards - self.mean) / self.std
        return rewards

class Reward3D_(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device
        self.blip       = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp        = MLP(768)
        self.Scorer     = Scorer()
        self.Scorer.eval()
        self.mean       = 0.16717362830052426
        self.std        = 1.0333394966054072

    def forward(self, image,prompt_ids, prompt_attention_mask):
        image_embeds = self.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt_ids,
                                                    attention_mask = prompt_attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
        view_features = text_output.last_hidden_state[:,0,:]       
        rw_features   = self.mlp(view_features,get_feature=128)    
        rw_features   = rw_features.view(rw_features.shape[0]//4, 4, 128)  
        rw_features   = torch.cat([rw_features[:, i, :] for i in range(4)], dim=1)
        rewards       = self.Scorer(rw_features)
        rewards = (rewards - self.mean) / self.std
        return rewards