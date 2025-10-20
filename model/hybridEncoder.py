# hybrid_encoder.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SSMBlock(nn.Module):
    """
    官方 Mamba-SSM 实现替代 MHA 中的局部依赖建模
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return self.ssm(x)


class HybridEncoderLayer(nn.Module):
    """
    每层HybridEncoder = SSM + MultiHeadAttention + FFN
    与原始 TransformerEncoderLayer 层数和串联方式一致
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ssm_block = SSMBlock(d_model, d_state, d_conv, expand)
        self.mha_block = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Step1: SSM局部依赖
        ssm_out = self.ssm_block(x)
        x = self.norm1(x + self.dropout(ssm_out))

        # Step2: Multi-head Attention全局依赖
        attn_out, _ = self.mha_block(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm2(x + self.dropout(attn_out))

        # Step3: FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class HybridEncoder(nn.Module):
    """
    堆叠多层HybridEncoderLayer
    与原始TransformerEncoderLayer层数保持一致（比如两层串联）
    """
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x
