import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.transformer2.conformer import Wav2Vec2ConformerEncoder as FlashConformerEncoder
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder as HFConformerEncoder,
)
from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
)

WIDTH_REDUCTION = 4


class Res2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 2), kernel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x + identity


class Conv2dSubsampling(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, input_height=128):
        super().__init__()
        self.module1 = Res2dModule(in_channels, out_channels, stride=(2, 2))
        self.module2 = Res2dModule(out_channels, out_channels, stride=(2, 2))
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, 1)
            dummy = self.module1(dummy)
            dummy = self.module2(dummy)
            freq_out = dummy.shape[2]
        self.linear = nn.Linear(out_channels * freq_out, 1024)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(B, W, C * H)
        x = self.linear(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, is_flash=False):
        super().__init__()
        self.is_flash = is_flash
        self.conv = Conv2dSubsampling(in_channels=in_channels, out_channels=512)
        self.proj = nn.Linear(1024, 256)

        conformer_config = Wav2Vec2ConformerConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=12,
            intermediate_size=4096,
            conv_depthwise_kernel_size=31,
        )

        if is_flash:
            self.conformer = FlashConformerEncoder(
                hidden_size=1024,
                num_heads=16,
                num_layers=12,
                intermediate_size=4096,
                conv_kernel=31,
            )
        else:
            self.conformer = HFConformerEncoder(conformer_config)

    def forward(self, x):
        x = self.conv(x)
        out = self.conformer(x) if self.is_flash else self.conformer(x)
        if isinstance(out, torch.Tensor):
            x = out
        else:
            x = out[0]
        x = self.proj(x)
        return x

    def load_pretrained(self, path):
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            k = k[6:] if k.startswith("model.") else k
            if k.startswith("conv.") or k.startswith("conformer."):
                new_state[k] = v
        msg = self.load_state_dict(new_state, strict=False)
        print(f"Loaded MSD pretrained weights: {msg}")
