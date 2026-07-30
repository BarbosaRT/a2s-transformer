import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2Vec2ConformerSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.max_distance = max_distance

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.pos_bias_embed = nn.Embedding(max_distance * 2, self.num_heads)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        B, L, C = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        pos_bias = self._compute_pos_bias(L, hidden_states.device)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            pos_bias_4d = pos_bias.permute(2, 0, 1).unsqueeze(0)
            attn_weights = attn_weights + pos_bias_4d
            return (attn_output, attn_weights)

        return attn_output

    def _compute_pos_bias(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance = distance.clamp(-self.max_distance, self.max_distance) + self.max_distance
        pos_bias = self.pos_bias_embed(distance)
        return pos_bias.permute(2, 0, 1)


class Wav2Vec2ConformerFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2ConformerConvModule(nn.Module):
    def __init__(self, hidden_size, conv_kernel=31, dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pointwise_conv1 = nn.Conv1d(
            hidden_size, hidden_size * 2, kernel_size=1, stride=1, padding=0
        )
        self.glu_activation = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=conv_kernel,
            stride=1,
            padding=conv_kernel // 2,
            groups=hidden_size,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()
        self.pointwise_conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, stride=1, padding=0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu_activation(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2ConformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, conv_kernel=31, dropout=0.0):
        super().__init__()
        self.self_attn = Wav2Vec2ConformerSelfAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn1 = Wav2Vec2ConformerFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=dropout,
        )
        self.conv_module = Wav2Vec2ConformerConvModule(
            hidden_size=hidden_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )
        self.ffn2 = Wav2Vec2ConformerFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=dropout,
        )
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_ffn1 = nn.LayerNorm(hidden_size)
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.norm_ffn2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.self_attn(self.norm_attn(hidden_states), attention_mask)
        hidden_states = hidden_states + 0.5 * self.ffn1(self.norm_ffn1(hidden_states))
        hidden_states = hidden_states + self.conv_module(self.norm_conv(hidden_states))
        hidden_states = hidden_states + 0.5 * self.ffn2(self.norm_ffn2(hidden_states))
        return hidden_states


class Wav2Vec2ConformerEncoder(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_layers=12, intermediate_size=4096, conv_kernel=31, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Wav2Vec2ConformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                conv_kernel=conv_kernel,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
