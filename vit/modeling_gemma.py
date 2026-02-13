import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

# TODO: Implement KV Cache
class KVCache():

class GemmaConfig():
    # Typical Decoder config
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000, 
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index # placeholder token id for image placeholder
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim # Output size of last linear projector of vision encoder
        self.hidden_size = hidden_size # hidden size of language model
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
        
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps # To avoid division by zero
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # Reciprocal sqrt -> 1 / sqrt
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float()) 
        return output.type_as(x)  

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # gate and up expands the hidden states to intermediate size, down projects it back down to hidden size
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class GemmaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_Dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0 # hidden size is split into multi heads

        # Grouped Query Attention:
        # Suppose:
        # hidden_size = 1024
        # num_head = 8
        # head_dimm = 1024 / 8 = 1024
        # Wq -> [1024, 8 * 128] -> [1024, 1024]
        # Wk -> [1024, 1 * 128] -> [1024, 128] # Every 8 heads of the q will share 1 head of the key, adjusted according to num_kv_heads
        # Wv -> [1024, 1 * 128] -> [1024, 128]
        # Less heads for k and v, reducing KV Cache overhead, resulting in lower data transfer between RAM <-> GPU
        # Compromises quality slightly for big performance gains
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            kv_cache=None,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # Transpose because we want to split the computation into groups of token sequences -> N, headsQ or headsKV, seq_len, head_dim
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rope
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            # Updates the KVCache at the same time preparing for the current pass
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx) 



class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
        # N, t, d
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # N, t, d
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        # N, t, d
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # N, t, d
        return hidden_states 


class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_sizer, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(self,
                attention_mask=None,
                position_ids=None,
                input_embeds=None,
                kv_cache=None
            ): 
        # N, t, d
        hidden_states = input_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        # N, t, d
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache =kv_cache
            )
        # N, t, d
        hidden_states = self.norm(hidden_states) # RMSNorm
        # N, t, d
        return hidden_states

class GemmaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask=None,
            poisition_ids=None,
            input_embeds=None,
            kv_cache=None,
    ) -> Tuple:
        # input embeds: N, t, d
        # output: N, t, d
        outputs = self.model(
            attention_mask=attention_mask,
            poisition_ids=poisition_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModelProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        # N, num_patches, embed_dim => N, num_patches, projection_dim
        hidden_states = self.linear(image_features)
        return hidden_states

# Class that connects all modality components (text + image)
class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModelProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        # shares embedding table and lm_head weights
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, image_features, input_embeds, input_ids, attention_mask, kv_cache
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        # Shape: N, seq len, hidden size\
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)

        # Create mask to determine which token is a placeholder token
        # Example:
        # Prompt:    [546, 546, 546, 546, 546, 546, 1, 234, 54, 63, 23, 35]
        # Text mask: [0, 0, 0, 0, 0, 0, 0, 234, 54, 63, 23, 35]
        # Image mask: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # Pad mask: all zeros as we dont use padding for now 
        # N, seq len -> True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # N, seq len -> True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # N, seq len -> True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # Expand the masks so we can use them with torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add text embeddings
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding) # (condition? true, false)
        # Insert image embeddings, cant use torch.where as scaled_image_features has different seq length
        final_embedding = final_embedding.mask_scatter(image_mask_expanded, scaled_image_features) # Where mask is true, copy from scaled image features
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####
        # Paligemma does not mask out future tokens in prefill mode (Image and prompt tokens attend to future tokens)

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            ) # q_len is 1, kv_len is the length of the previous tokens, hence no masking, since need to contextualize on all prev tokens. 

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # Does not mask out image tokens
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1] # [0, 1, 2, ... number of tokensin kv cache + 1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
    
    def forward(self, input_ids, pixel_values, attention_mask, kv_cache: Optional[KVCache] = None):
        assert torch.all(attention_mask == 1), "The input cannot be padded."

        # 1. Extract the input embeddings
        # Batch_size, seq_len, hidden_size
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # N, C, H, W -> N, num patches, embed dim
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # N, num patches, embed dim -> N, num patches, hidden size, resize to that it matches text hidden size so they can be concatenated
        image_features = self.multimodal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens, replace all placeholder tokens with real image feature embeddings
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs