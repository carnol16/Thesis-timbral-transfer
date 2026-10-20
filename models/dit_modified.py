import torch.nn as nn 
import torch
import math

LATENT_DIM = 256 #SAME latent channels
CONTOUR_DIM = 6 # log_f0, voiced, rms, onset, aperiodicity, centroid
SEQ_LEN = 32 # latent frames per clip
N_INSTRUMENTS = 18 #number of instruments in the dataset : we are using the violin (#7)

D_MODEL = 768 #model width
DEPTH = 16 #number of DiT blocks
N_HEADS = 12

HEAD_DIM = D_MODEL // N_HEADS
USE_QKNORM = True
ROPE_THETA = 10000.0

def timestep_embedding(t, dim, max_period = 10000):
    #Sinusoidal embedding of ascalar timestep => (B,dim)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device = t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim = -1)


def modulate(x, shift, scale):
        # x: (B, T, D);  shift/scale: (B, D) -> broadcast over the time axis
    return x * (1+scale[:, None]) + shift[:, None]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        # Calculate RMS (Root Mean Square)
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / norm
        if self.elementwise_affine:
            return x_normed * self.weight
        return x_normed


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # Typical SwiGLU hidden dim is 8/3 of the model dim
            hidden_dim = int(dim * 8 / 3)
            
        # Two parallel linear layers for the gate and the value
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        # Output projection
        self.w3 = nn.Linear(hidden_dim, dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        # Element-wise multiplication of the activated gate (w1) and value (w2)
        return self.w3(self.silu(self.w1(x)) * self.w2(x))


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  
    freqs = torch.outer(t, freqs).float()  
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq.shape[-1] // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True) if USE_QKNORM else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True) if USE_QKNORM else nn.Identity()
        
    def forward(self, x, freqs_cis=None):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim) for t in qkv]
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.proj(attn_out)


class DITBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        
        self.norm1 = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = SelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(dim)

        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)  
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, freqs_cis=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1) 

        # Attention Sub-layer
        h = modulate(self.norm1(x), shift1, scale1)
        attn_out = self.attn(h, freqs_cis=freqs_cis) 
        x = x + gate1[:, None] * attn_out

        # MLP Sub-layer
        h2 = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h2)
        x = x + gate2[:, None] * mlp_out
        
        return x


class ContourRendererDiT(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, contour_dim=CONTOUR_DIM, d_model=D_MODEL, depth=DEPTH, n_heads=N_HEADS, contour_drop_prob=0.0, use_rope=False, use_instrument_embed=False, num_instruments=N_INSTRUMENTS):
        super().__init__()
        self.x_proj = nn.Linear(latent_dim + contour_dim, d_model)
        self.contour_dim = contour_dim
        self.t_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([DITBlock(dim = d_model, n_heads=n_heads) for _ in range(depth)])
        self.contour_drop_prob = contour_drop_prob
        self.d_model = d_model
        self.use_instrument_embed = use_instrument_embed
        if self.use_instrument_embed:
            self.instrument_embed = nn.Embedding(num_instruments, d_model)

        self.use_rope = use_rope
        
        # Absolute positional embedding fallback if RoPE is not used
        pos_embed = timestep_embedding(torch.arange(SEQ_LEN), d_model)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        
        self.final_norm = RMSNorm(d_model, elementwise_affine=False, eps=1e-6) 

        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_ada[1].weight)   
        nn.init.zeros_(self.final_ada[1].bias)
        
        self.proj_out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.proj_out.weight)   
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, t, contours, instrument_id=None, start_pos=0):
        c = self.t_mlp(timestep_embedding(t, self.d_model))
        
        # CFG: Generate dropout mask first so we can drop both contour AND instrument
        if self.training and self.contour_drop_prob > 0:
            drop_mask = torch.rand(x.shape[0], 1, 1, device=x.device) < self.contour_drop_prob
            contours = contours * (~drop_mask).float()
            
            # Add instrument embedding, but zero it out if drop_mask is True
            if self.use_instrument_embed and instrument_id is not None:
                inst_emb = self.instrument_embed(instrument_id)
                # drop_mask is [B, 1, 1]. inst_emb is [B, d_model]
                # Squeeze drop_mask to [B, 1] for broadcasting
                inst_emb = inst_emb * (~drop_mask.squeeze(-1)).float()
                c = c + inst_emb
        else:
            # During inference or no CFG
            if self.use_instrument_embed and instrument_id is not None:
                c = c + self.instrument_embed(instrument_id)

        x = self.x_proj(torch.cat([x,contours],dim = -1))

        if self.use_rope:
            seq_len = x.shape[1]
            freqs_cis = precompute_freqs_cis(HEAD_DIM, start_pos + seq_len, theta=ROPE_THETA).to(x.device)
            freqs_cis = freqs_cis[start_pos:]
        else:
            # Use APE if RoPE is disabled
            x = x + self.pos_embed
            freqs_cis = None

        for block in self.blocks:
            x = block(x, c, freqs_cis=freqs_cis)

        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(x), shift, scale)
        final_out = self.proj_out(h)

        return final_out


#Smoke test
if __name__ == "__main__":
    model = ContourRendererDiT(use_rope=True)
    x = torch.randn(2, SEQ_LEN, LATENT_DIM)
    t = torch.randint(0, 1000, (2,))
    contours = torch.randn(2, SEQ_LEN, CONTOUR_DIM)

    out = model(x, t, contours)

    print("Smoke Test Passed!")
    print(f"Output shape: {out.shape}")
