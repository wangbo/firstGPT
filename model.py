import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LLMOutput:
    logits: torch.Tensor = None

class LayerNorm(nn.Module):

  def __init__(self, ctx):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ctx.embedding_dim))
    self.bias = nn.Parameter(torch.zeros(ctx.embedding_dim)) if ctx.bias else None

  def forward(self, x):
    return F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, 1e-5)

class MultiHeadAttn(nn.Module):
  def __init__(self, ctx):
    super().__init__()
    self.block_size = ctx.block_size
    self.embedding_dim = ctx.embedding_dim
    self.head_num = ctx.head_num
    assert self.embedding_dim % self.head_num == 0
    self.head_dim = ctx.embedding_dim // self.head_num

    self.qkv_pro = nn.Linear(ctx.embedding_dim, self.embedding_dim * 3, bias=ctx.bias)
    self.output_pro = nn.Linear(self.embedding_dim, self.embedding_dim, bias=ctx.bias)

    self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size))
                         .view(1,1,self.block_size,self.block_size))
    self.attn_dropout = nn.Dropout(ctx.dropout)
    self.residual_dropout = nn.Dropout(ctx.dropout)

    self.dropout = ctx.dropout
    self.flash_attn = (hasattr(torch.nn.functional, 'scaled_dot_product_attention')) if ctx.enable_flash else None

  def forward(self, x):
    B,T,C = x.shape

    q,k,v = self.qkv_pro(x).split(self.embedding_dim, dim=2)
    q = q.view(B,T,self.head_num,self.head_dim).transpose(1,2)
    k = k.view(B,T,self.head_num,self.head_dim).transpose(1,2)
    v = v.view(B,T,self.head_num,self.head_dim).transpose(1,2)

    if self.flash_attn:
      output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                dropout_p=self.dropout if self.training else 0,
                                                                is_causal=True)
    else:
      attn_score = q @ k.transpose(2,3) / math.sqrt(self.head_dim)
      attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
      attn_score = F.softmax(attn_score, dim=-1)
      attn_score = self.attn_dropout(attn_score)
      output = attn_score @ v # (B,head_num,T,T) @ (B,head_num,T,head_dim)

    output = output.transpose(1,2).contiguous().view(B,T,C)

    output = self.residual_dropout(self.output_pro(output))
    return output

class MLP(nn.Module):

  def __init__(self, ctx):
    super().__init__()

    self.up_pro = nn.Linear(ctx.embedding_dim, ctx.embedding_dim * 4, bias=ctx.bias)
    self.gelu = nn.GELU()
    self.down_pro = nn.Linear(ctx.embedding_dim * 4, ctx.embedding_dim, bias=ctx.bias)
    self.dropout = nn.Dropout(ctx.dropout)

  def forward(self, x):
    x = self.up_pro(x)
    x = self.gelu(x)
    x = self.down_pro(x)
    x = self.dropout(x)
    return x

class GPTBlock(nn.Module):

  def __init__(self,ctx):
    super().__init__()
    self.ln_1 = LayerNorm(ctx)
    self.attn = MultiHeadAttn(ctx)
    self.mlp = MLP(ctx)
    self.ln_2 = LayerNorm(ctx)

  def forward(self,x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT2Model(nn.Module):

  def __init__(self,ctx):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=ctx.vocab_size, embedding_dim=ctx.embedding_dim)
    self.pos_embed = nn.Embedding(num_embeddings=ctx.block_size, embedding_dim=ctx.embedding_dim)
    self.dropout = nn.Dropout(ctx.dropout)
    self.layers = nn.ModuleList([GPTBlock(ctx) for idx in range(ctx.n_layer)])
    self.ln_f = LayerNorm(ctx)
    self.lm_head = nn.Linear(ctx.embedding_dim, ctx.vocab_size, bias=False)

    self.lm_head.weight = self.embedding.weight

    self.apply(self._init_weights)
    for pn,p in self.named_parameters():
      if pn.endswith("down_pro.weight") or pn.endswith("output_pro.weight"):
         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * ctx.n_layer))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self,input_ids):
    B,T = input_ids.shape
    input_embedding = self.embedding(input_ids)
    pos_embedding = self.pos_embed(torch.arange(0, T, dtype=torch.long, device=input_ids.device))
    hidden_state = self.dropout(input_embedding + pos_embedding)

    for decode_layer in self.layers:
      hidden_state = decode_layer(hidden_state)
    hidden_state = self.ln_f(hidden_state)

    logits = self.lm_head(hidden_state)
    return LLMOutput(logits=logits)


