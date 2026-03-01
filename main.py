from model import GPTContext, GPT2Model

ctx = GPTContext()
ctx.block_size = 256
ctx.vocab_size = 50000

ctx.num_hidden_layer = 12
ctx.embedding_size = 768
ctx.head_num = 12
ctx.head_dim = 64
ctx.mlp_hidden_size = 3072
model = GPT2Model(ctx)

# for p in model.parameters():
#     print(p.numel())

for name,p in model.named_parameters():
    print(f"{name: <50} {p.numel():>10,}")

param_num = sum(p.numel() for p in model.parameters())
print(f"{param_num/1e6}M parameters, param:{param_num}")

#161.968128M parameters, param:161968128
#161.968128M parameters, param:161968128
