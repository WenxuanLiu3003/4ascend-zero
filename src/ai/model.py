from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class PolicyValueNet(nn.Module):
    def __init__(self, in_planes: int, board_size: int = 9, width: int = 256, n_blocks: int = 8):
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(width) for _ in range(n_blocks)])
        # policy head
        self.p_conv = nn.Conv2d(width, 2, 1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        # value head
        self.v_conv = nn.Conv2d(width, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board_size * board_size, width)
        self.v_fc2 = nn.Linear(width, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.res(self.stem(x))
        # policy
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)  # logits over H*W
        # value
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))  # [-1, 1]
        return p, v.squeeze(-1)