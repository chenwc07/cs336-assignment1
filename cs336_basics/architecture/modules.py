import torch
from torch import nn
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor

from einops import rearrange, einsum, repeat

class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.device = device if device else torch.get_default_device()

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2.0/(in_features + out_features))
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, " ... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.device = device if device else torch.get_default_device()

        embeddings_matrix = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        std = 2.0 / math.sqrt(num_embeddings + embedding_dim)
        nn.init.trunc_normal_(embeddings_matrix, mean=0.0, std=std, a=-3*std, b=3*std)
        self.embeddings = nn.Parameter(embeddings_matrix)

    def forward(
            self, 
            token_ids: Int[Tensor, " ..."]
    ) -> torch.Tensor:
        return self.embeddings[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.device = device if device else torch.get_default_device()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # rms_norm
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms_x * self.g

        return x_normed.to(in_dtype)

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)

class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.device = device if device else torch.get_default_device()
        # d_ff = round(8 / 3 * d_model / 64) * 64
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.fc3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.fc2(silu(self.fc1(x)) * self.fc3(x))
    

class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device if device else torch.get_default_device()
        self.d_k = d_k

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()

        freqs = einsum(positions, inv_freq, "i, j -> i j")  # (max_seq_len, d_k/2)
        
        emb = repeat(freqs, "i j -> i (j 2)")  # (max_seq_len, d_k)

        self.register_buffer("cos_emb", emb.cos())  # (max_seq_len, d_k)
        self.register_buffer("sin_emb", emb.sin())  # (max_seq_len, d_k)

    def forward(
            self,
            x: Float[Tensor, " ... seq_len d_k"],
            token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos_emb = self.cos_emb[token_positions]  # (..., seq_len, d_k)
        sin_emb = self.sin_emb[token_positions]  # (..., seq_len, d_k)

        x1 = x[..., 0::2] # (..., seq_len, d_k/2)
        x2 = x[..., 1::2] # (..., seq_len, d_k/2)
        x_rotated = rearrange(torch.stack([-x2, x1], dim=-1), "... seq_len d_k_2 b -> ... seq_len (d_k_2 b)")  # (..., seq_len, d_k)

        return x * cos_emb + x_rotated * sin_emb

def softmax(x: Tensor, dim: int) -> Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(
        query: Float[Tensor, " ... seq_len_q d_k"],
        key: Float[Tensor, " ... seq_len_k d_k"],
        value: Float[Tensor, " ... seq_len_k d_v"],
        mask: Bool[Tensor, "... seq_len_q seq_len_k"] | None = None
) -> Float[Tensor, " ... seq_len_q d_v"]:
    d_k = query.shape[-1]
    logits = einsum(query, key, "... i d_k, ... j d_k -> ... i j") / math.sqrt(d_k) # (..., seq_len_q, seq_len_k)
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    logits = softmax(logits, dim=-1)  # (..., seq_len_q, seq_len_k)
    out = einsum(logits, value, "... i j, ... j d_v -> ... i d_v")
    return out

class CasualMutiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        # rope: RoPE | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.proj_q = Linear(d_model, d_model)  # (d_model, num_heads * d_k)
        self.proj_k = Linear(d_model, d_model)  # (d_model, num_heads * d_k)
        self.proj_v = Linear(d_model, d_model)  # (d_model, num_heads * d_k)
        self.out_proj = Linear(d_model, d_model)  # (num_heads * d_k, d_model)
        # if rope:
        #     self.rope = rope

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        rope: RoPE | None = None,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        q = self.proj_q(x)  # (..., seq_len, num_heads * d_k)
        k = self.proj_k(x)  # (..., seq_len, num_heads * d_k)
        v = self.proj_v(x)  # (..., seq_len, num_heads * d_k)

        q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., num_heads, seq_len, d_k)
        k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., num_heads, seq_len, d_k)
        v = rearrange(v, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., num_heads, seq_len, d_k)

        # if hasattr(self, 'rope') and token_positions is not None:
        if rope is not None and token_positions is not None:
            q = rope(q, token_positions)  # (..., num_heads, seq_len, d_k)
            k = rope(k, token_positions)  # (..., num_heads, seq_len, d_k)

        # casual mask
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()  # (seq_len, seq_len)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask
        )  # (..., num_heads, seq_len, d_k)

        out = rearrange(out, "... h seq_len d_k -> ... seq_len (h d_k)")  # (..., seq_len, num_heads * d_k)
        out = self.out_proj(out)  # (..., seq_len, d_model)

        return out
        
class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
    ):
        super().__init__()
        self.attn = CasualMutiHeadSelfAttention(d_model, num_heads)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        rope: RoPE | None = None,
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        if not token_positions:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        x = x + self.attn(self.ln1(x), rope=rope, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=context_length)

    def forward(
        self,
        in_indices: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len vocab_size"]:
        x = self.token_embeddings(in_indices) # (..., seq_len, d_model)
        for layer in self.layers:
            x = layer(x, rope=self.rope)
        x = self.ln_final(x) # (..., seq_len, d_model)
        x = self.lm_head(x) # (..., seq_len, vocab_size)
        return x
    