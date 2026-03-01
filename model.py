import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):

  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

    self.embedding = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))

  def forward(self, x):
    output = self.embedding[x]
    return output

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

  def forward(self, x):
    B,T,C = x.shape

    query,key,value = self.qkv_pro(x).split(self.embedding_dim, dim=2)
    query = query.view(B,T,self.head_num,self.head_dim).transpose(1,2)
    key = key.view(B,T,self.head_num,self.head_dim).transpose(1,2)
    value = value.view(B,T,self.head_num,self.head_dim).transpose(1,2)

    attn_score = query @ key.transpose(2,3) / math.sqrt(self.head_dim)
    attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
    attn_score = F.softmax(attn_score, dim=-1)
    attn_score = self.attn_dropout(attn_score)

    output = attn_score @ value # (B,head_num,T,T) @ (B,head_num,T,head_dim)
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

class LayerNorm(nn.Module):

  def __init__(self,ctx):
    super().__init__()
    self.eps = 1e-5

    self.weight = nn.Parameter(torch.ones(ctx.embedding_dim))
    self.bias = nn.Parameter(torch.zeros(ctx.embedding_dim)) if ctx.bias else None

  def forward(self, x):
    mean_val = x.mean(dim=-1, keepdim=True)
    var_val = x.var(dim=-1,keepdim=True,unbiased=False)
    x = (x - mean_val) / (var_val + self.eps).sqrt()
    x = x * self.weight
    if self.bias is not None:
      x = x + self.bias
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
    self.embed = EmbeddingLayer(ctx.vocab_size,ctx.embedding_dim)
    self.pos_embed = EmbeddingLayer(ctx.block_size,ctx.embedding_dim)
    self.dropout = nn.Dropout(ctx.dropout)
    self.layers = nn.ModuleList([GPTBlock(ctx) for idx in range(ctx.n_layer)])
    self.ln_f = LayerNorm(ctx)
    self.lm_head = nn.Linear(ctx.embedding_dim, ctx.vocab_size, bias=False)

    self.lm_head.weight = self.embed.embedding

    self.apply(self._init_weights)
    for pn,p in self.named_parameters():
      if pn.endswith("down_pro.weight"):
         print(pn)
         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * ctx.n_layer))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, EmbeddingLayer):
        torch.nn.init.normal_(module.embedding, mean=0.0, std=0.02)

  def forward(self,x):
    B,T = x.shape
    input_embedding = self.embed(x)
    pos_embedding = self.pos_embed(torch.arange(0, T, dtype=torch.long, device=x.device))
    hidden_state = self.dropout(input_embedding + pos_embedding)

    for decode_layer in self.layers:
      hidden_state = decode_layer(hidden_state)
    hidden_state = self.ln_f(hidden_state)

    output = self.lm_head(hidden_state)
    return output

  def generate(self, token_ids, max_length):
    token_ids = torch.tensor(token_ids)
    token_ids = token_ids.reshape(1, *token_ids.shape)

    while len(token_ids[0]) < max_length:
      output = self.forward(token_ids)
      output = F.softmax(output,dim=-1)

      new_token_id = torch.multinomial(output[0][-1],num_samples=1, replacement=False)
      new_token_id = torch.tensor([[new_token_id]])
      token_ids = torch.cat((token_ids, new_token_id),dim=-1)
    return token_ids[0].tolist()