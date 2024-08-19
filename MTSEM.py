from torch import nn
import torch

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, in_channels, heads=3):
        super().__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.attention_heads = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
            for _ in range(heads)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        features = torch.cat([torch.max(x, dim=1, keepdim=True)[0],
                              torch.mean(x, dim=1, keepdim=True)], dim=1)
        attention_maps = [self.sigmoid(head(features)) for head in self.attention_heads]
        attended_features = [attention_map * x for attention_map in attention_maps]
        combined_features = torch.cat(attended_features, dim=1)
        output = torch.mean(combined_features, dim=1, keepdim=True)
        return output

class MT_SEM(nn.Module):
    def __init__(self, kernel_size=7, num_tasks=3, heads_per_task=8):
        super().__init__()
        assert kernel_size in (3, 7)
        self.attention_blocks = nn.ModuleList([
            MultiHeadAttentionModule(2, heads=heads_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x):
        input = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        outputs = []
        for block in self.attention_blocks:
            attention = block(x)
            outputs.append(attention * input)
        return outputs