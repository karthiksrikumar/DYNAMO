import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, t):
        """
        t: (batch_size, 2) [year, month]
        returns: (batch_size, embed_dim)
        """
        batch_size = t.shape[0]
        output = torch.zeros(batch_size, self.embed_dim).to(t.device)
        
        # Periodic components
        for k in range(1, (self.embed_dim // 2) + 1):
            freq = 1 / (10 ** (4 * k / self.embed_dim))
            output[:, 2*k-2] = torch.sin(2 * math.pi * t[:, 0] * freq)
            output[:, 2*k-1] = torch.cos(2 * math.pi * t[:, 0] * freq)
        
        # Linear component for month
        if self.embed_dim % 2 == 1:
            output[:, -1] = t[:, 1] / 12.0  # Normalize month
        
        return output

class CausalTemporalAdapter(nn.Module):
    def __init__(self, hidden_size, time_embed_dim, causal_embed_dim, rank=8):
        super().__init__()
        self.time_embed = Time2Vec(time_embed_dim)
        
        # Time projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, rank),
            nn.GELU(),
            nn.Linear(rank, hidden_size)
        )
        
        # Causal projection
        self.causal_proj = nn.Sequential(
            nn.Linear(causal_embed_dim, rank),
            nn.GELU(),
            nn.Linear(rank, hidden_size)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, time_input, causal_input):
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        time_input: (batch_size, 2) [year, month]
        causal_input: (batch_size, causal_embed_dim)
        """
        # Process time and causal inputs
        time_embed = self.time_embed(time_input)  # (batch_size, time_embed_dim)
        time_repr = self.time_proj(time_embed).unsqueeze(1)  # (batch_size, 1, hidden_size)
        causal_repr = self.causal_proj(causal_input).unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Combine time and causal information
        combined = time_repr + causal_repr
        
        # Gated residual connection
        gate = self.gate(torch.cat([hidden_states, combined.expand_as(hidden_states)], dim=-1))
        output = hidden_states + gate * combined
        return self.norm(output)
