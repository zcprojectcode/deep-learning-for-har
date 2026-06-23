"""
Shavit and Klein, Boosting Inertial-based Human Activity Recognition with Transformers 
IEEE Open Access (2021)
https://doi.org/10.1109/ACCESS.2021.3070646   

MIT License

Copyright (c) 2026 yolish

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
""" 

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class IMUTransformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dimension = config.dimension
        self.window_size = config.window

        self.project_input = nn.Sequential(nn.Conv1d(config.features, self.dimension, 1), nn.GELU(),
                                        nn.Conv1d(self.dimension, self.dimension, 1), nn.GELU(),
                                        nn.Conv1d(self.dimension, self.dimension, 1), nn.GELU(),
                                        nn.Conv1d(self.dimension, self.dimension, 1), nn.GELU())

        encoder = TransformerEncoderLayer(d_model = self.dimension,
                                            nhead = config.heads,
                                            dim_feedforward = config.dim_feedforward,
                                            dropout = config.dropout,
                                            activation = "gelu")

        self.transformer_encoder = TransformerEncoder(encoder,
                                                    num_layers = config.depth,
                                                    norm = nn.LayerNorm(self.dimension))
                                                    
        self.token = nn.Parameter(torch.zeros((1, self.dimension)), requires_grad=True)

        self.positional_embedding = nn.Parameter(torch.randn(self.window_size + 1, 1, self.dimension))

        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.dimension),
            nn.Linear(self.dimension,  self.dimension//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.dimension//4, config.classes)
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        imu_data = data.get("iner")

        # Transform to higher dimensional space
        imu_data = self.project_input(imu_data.transpose(1, 2)).permute(2, 0, 1)

        token = self.token.unsqueeze(1).repeat(1, imu_data.shape[1], 1) # Prepend class token
        imu_data = torch.cat([token, imu_data])
        imu_data += self.positional_embedding

        target = self.transformer_encoder(imu_data)[0]
        target = self.log_softmax(self.imu_head(target))

        return target
