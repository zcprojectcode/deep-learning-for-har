"""
Lamaakal, I., Yahyati, C., Maleh, Y. et al. 
A tiny inertial transformer for human activity recognition via multimodal knowledge 
distillation and explainable AI. Sci Rep 15, 42335 (2025). 
https://doi.org/10.1038/s41598-025-26297-2

MIT License

Copyright (c) 2025 Ismail Lamaakal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/Ism-ail11/XTinyHAR/tree/main?tab=MIT-1-ov-file
"""

import torch
import torch.nn as nn
from torch import Tensor
from .blocks import PatchEmbed1D, TransformerEncoderLayer, PositionalEmbeddingLearned

class TinyTransformer(nn.Module):
    def __init__(self, in_channels=6, window_size=100, patch_size=20, dim=128, depth=2, heads=4, mlp_ratio=2.0, num_classes=27, drop=0.1):
        super().__init__()
        self.patch = PatchEmbed1D(in_channels, patch_size, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = PositionalEmbeddingLearned(seq_len=(window_size // patch_size) + 1, dim=dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: Tensor):
        x = self.patch(x)  # (B, N, D)
        B, N, D = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        x = self.norm(x)
        cls_tok = x[:, 0]
        logits = self.head(cls_tok)
        return logits, attn_maps