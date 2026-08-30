from transformers import PretrainedConfig


class StanMindConfig(PretrainedConfig):
    model_type = "stanmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch.nn import functional as F
from .activation_fuctions import ACT2FN

# 继承 nn.Module类
class RMSNorm(nn.Module):
# __init__ 初始化
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
# _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
# forward 
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim:int, end:int(32*1024), rope_base, rope_scaling:Optional[dict] = None):
    # 初始化rope频率
    freqs, attn_factor = 1 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)), 1.0

    if rope_scaling is not None:
        orgin_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"]
        )

    # 如果推断的长度 大于 训练长度，使用 YaRN 进行缩放
    if end > orgin_max:
        # 波长b到i的映射
        inv_dim = lambda b : (dim * math.log(orgin_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        # 划分高低维度
        # low 不需要缩放的 高频部分
        # high 需要缩放的 低频部分
        low ,high = max(math.floor(inv_dim(beta_fast)), 0) , min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

        # 计算缩放因子
        # low部分 ramp = 0, high部分 ramp = 1， low与high之间平滑过渡
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low)
            / max(high - low, 0.001),
            0,
            1,
        )

        # 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
        # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
        # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
        # ramp在0-1之间时：平滑过渡。
        freqs = freqs * (1 - ramp + ramp / factor)

    #根据end，计算位置索引 t
    t = torch.arange(end, device=freqs.device).float()

    # 计算频率和位置的外积，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

# 编写RoPE的应用函数
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [a, b] -> [-b, a]
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
    #x_rotated=x*cos+rotate_ half(×)*sin   
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x : torch.Tensor, n_rep : int)-> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args : StanMindConfig):
        super().__init__()

        # 得到KV 的 头数
        if args.num_key_value_heads:
            self.num_key_value_heads = args.num_key_value_heads
        else:
            self.num_key_value_heads = args.num_attention_heads  

        # 确保Q 的头数 可以 整除 kv的头数
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 对线性层的 QKV进行一个定义
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)

        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias = False)

        # 定义变量
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 计算加速
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention


    def forward(self, 
                x : torch.Tensor, 
                position_embedding : Tuple[torch.Tensor, torch.Tensor],
                past_key_value : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                use_cache = False, 
                attention_mask : Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # 投影，计算q k v 
        bsz, seq_len, _ = x.shape
        xq,xk,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 用 view， 把输入拆分成多个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # q k 使用 rope
        cos, sim = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sim[:seq_len])
    
        # k v 使用 repeat， kv cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_key_value = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            # [bsz, self.n_local_heads, seq_len, self.head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2), 
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 注意力公式计算
        # pytorch内置实现
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask = attn_mask, 
                                                    dropout_p = self.dropout if self.training else 0.0, is_causal = True)
        else:
            # 自定义实现
            scores = (xq@xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len,seq_len), float('-inf'), device = scores.device), diagonal = 1
                ).unsqueeze(0).unsqueeze(0)

        # 最后拼接多头结果，放回
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            scores = scores + extended_attention_mask

        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        scores = self.attn_dropout(scores) 
        output = scores@xv
        # [bsz, self.n_local_heads, seq_len, self.head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_key_value


class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 门控
    # 激活函数
    # 降维
    # dropout
    def __init__(self, args : StanMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias = False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias = False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias = False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x): 
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class StanMindBlock(nn.Module):
    def __init__(self, layer_id : int, config : StanMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embedding, past_key_value = None, use_cache = False, attention_mask = None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embedding, past_key_value, use_cache, attention_mask
        )
        # 残差处理
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

    