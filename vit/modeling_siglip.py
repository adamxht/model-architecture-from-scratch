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
    
class SiglipAttention(nn.Module):
    """ MHA based on the paper Attention is all you need """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        # No shape changes, just a transformation from embeddings to qkv
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_startes = N, num_patches (t), embed_dim (d)
        batch_size, seq_len, _ = hidden_states.size()
        # N, t, d -> N, t, d 
        q_states = self.q_proj(hidden_states) # Wq
        # N, t, d -> N, t, d
        k_states = self.k_proj(hidden_states) # Wk
        # N, t, d -> N, t, d
        v_states = self.v_proj(hidden_states) #Wv

        # Splitting into multiple heads: N, t, d -> N, d/headDim, t. d/numHeads
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Now we can iterate through dim 1 (each head) to compute the attn weights
        # [N, NH, t, HD] * [N, NH, HD, t] -> [N, NH, t, t]
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) * self.scale
        # Bidirectional, so no causal mask

        # Apply softmax to each row of tokens/patch, last dimension
        attn_weights = nn.functional.Softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype) 
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # Not used, just a passthrough

        # [N, NH, t, t] * [N, NH, t, HD] -> [N, NH, t, HD]
        attn_output = torch.matmul(attn_weights, v_states)
        # Concat back all the heads: [N, NH, t, HD] -> [N, t, NH, HD]
        attn_output = hidden_states.transpose(1, 2).contiguous() # Make the tensor contiguous in memory, so that the next reshape will have no overhead
        # [N, t, NH, HD] -> [N, t, d]
        attn_output = hidden_states.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(hidden_states)
        return attn_output, attn_weights # Don't really need attn_weights for bidirectional encoders
        
    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) # Expanded into intermediate size, adding non linearity
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size) 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [N, num_patches, embed_dim] -> [ N, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [N, num_patches, intermediate_size] -> [ N, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh') # Non linearity
        # [N, num_patches, intermediate_size] -> [ N, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_sates: torch.Tensor) -> torch.Tensor:
        # Refer to VisionEncoder.png for architecture
        # [N, num_patches, embed_dim]
        residual = hidden_states
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        hidden_states = hidden_states + residual
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        residual = hidden_states # Another skip connection
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states) # Adds non linearity, adds degree of freedom
        # [N, num_patches, embed_dim] -> [ N, num_patches, embed_dim]
        hidden_states = hidden_states + residual
        return hidden_states


class SiglipVisionTransfmer(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # Convolution to image + positional encodings
        self.layers = nn.ModuleList(
            [SiglipVisionEncoder(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values) -> Tuple:
        hidden_states = self.embeddings(pixel_values=pixel_values)
        for layer in self.layers:
            # N, t, d
            hidden_states = self.layers(hidden_states=hidden_states)
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