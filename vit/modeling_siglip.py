from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.embed_dim = config.hidden_size

        # Input is channels, output is embedding dim, which means converting rgb to embedding space
        #W Since stride = kernel_size, non-overlapping patches
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, 
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        ) 
        num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim) # Learned positional lookup embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [N, C, H, W]

        patch_embeds = self.patch_embedding(pixel_values) # [N, Embed_dim, num_patches_h, num_patches_w] -> Convolved to patches' embeddings
        embeddings = patch_embeds.flatten(2) # [N, Embed_dim, num_patches]
        embeddings = embeddings.transpose(1, 2) # [N, num_patches, Embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids) # Add positional embeddings
        return embeddings # [N, num_patches, Embed_dim]

class SiglipVisionEncoder(nn.Module):
    pass

class SiglipVisionTransfmer(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # Convolution to image + positional encodings
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values) -> Tuple:
        hidden_states = self.embeddings(pixel_values=pixel_values)
        last_hidden_states = self.encoder(hidden_states=hidden_states)
        last_hidden_states = self.post_layernorm(last_hidden_states)
        return last_hidden_states

class SiglipVisionModel(nn.Module):
    
    def __init__(self, condfig: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        # [N, C, H, W] -> [N, Num_patches, Embed_dim]
        return self.vision_model(pixel_values=pixel_values)