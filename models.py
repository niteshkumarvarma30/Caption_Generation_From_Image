# models.py
import torch
import torch.nn as nn
from torchvision import models
from transformers import GPT2LMHeadModel

class XAICaptioner(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Pretrained ResNet-50 extracts spatial features
        resnet = models.resnet50(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze encoder to maintain stable visual features
        for param in self.encoder.parameters():
            param.requires_grad = False 

        # Bridge: Connects 2048-dim CNN features to 768-dim GPT-2 tokens
        self.bridge = nn.Linear(2048, 768)
        self.vis_norm = nn.LayerNorm(768) 
        
        # --- NEW: Visual Positional Encoding (Matches the Paper) ---
        # 49 patches (7x7 grid), each mapped to 768 dimensions
        self.vis_pos_embed = nn.Parameter(torch.zeros(1, 49, 768))
        nn.init.trunc_normal_(self.vis_pos_embed, std=0.02)
        
        # Decoder: GPT-2 with cross-attention enabled
        self.decoder = GPT2LMHeadModel.from_pretrained(
            "gpt2", 
            add_cross_attention=True,
            attn_implementation="eager" 
        )

        for name, module in self.decoder.named_modules():
            if "crossattention" in name.lower():
                for param in module.parameters():
                    param.data.normal_(mean=0.0, std=0.02)

    def forward(self, images, input_ids, masks):
        with torch.no_grad():
            features = self.encoder(images) 
        
        features = features.permute(0, 2, 3, 1).flatten(1, 2) 
        
        # Normalize and scale the visual signal
        encoder_hidden_states = self.vis_norm(self.bridge(features)) * 0.1 
        
        # --- NEW: Add the spatial coordinates to the visual features ---
        encoder_hidden_states = encoder_hidden_states + self.vis_pos_embed
        
        outputs = self.decoder(
            input_ids=input_ids, 
            attention_mask=masks, 
            encoder_hidden_states=encoder_hidden_states
        )
        
        outputs.logits = torch.clamp(outputs.logits, min=-50, max=50)
        return outputs