import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# works for 7 x 7 x 20 (H x W x C) images
class ImageEmbedding(nn.Module):
    def __init__(self, d_model):
        super(ImageEmbedding, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )   
        self.fc = nn.Sequential(
            nn.Linear(3920, d_model), # 3920 = 80 * 7 * 7
            nn.Tanh()
        )

    def forward(self, x):
        # x is of shape (batch_size, seq len, 3, 84, 84)
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(-1, 20, 7, 7)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        x = x.view(batch_size, seq_len, -1) # Reshape back to (batch_size, seq_len, d_model)
        return x
    
class ActionEmbedding(nn.Module):
    def __init__(self, d_model, action_dim):
        super(ActionEmbedding, self).__init__()
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.Tanh()
        )

    def forward(self, x):
        # x is of shape (batch_size, seq len)
        x = F.one_hot(x, num_classes=self.action_dim).float()
        x = self.mlp(x)
        return x
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max(10000, max_len) # 10000 allows for testing with larger sequence lengths than encountered during training

        # Compute the positional encodings in advance
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register the positional encodings as buffers (not learnable parameters)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional embeddings to the input
        return x + self.pe[:, :x.size(1)]

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, feed_forward_dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, d_model),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attention_output, attention_weights = self.self_attention(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + attention_output)
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)
        return x, attention_weights

class Transformer(nn.Module):
    def __init__(self, action_dim, d_model, nhead, num_layers, max_len, feed_forward_dim=2048):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model

        # Embedding layer to convert input tokens to d_model-dimensional vectors
        self.image_embedding = ImageEmbedding(d_model)
        self.action_embedding = ActionEmbedding(d_model, action_dim)
        self.positional_encoding = PositionalEmbedding(d_model, max_len)
        self.embedding_LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        # Transformer Encoder layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, nhead, feed_forward_dim) for _ in range(num_layers)])

        # Output layer
        self.output = nn.Linear(d_model, action_dim)

    def forward(self, obs, actions):
        batch_size, seq_len = obs.size(0), obs.size(1)

        # Embed the input tokens
        obs = self.image_embedding(obs)
        actions = self.action_embedding(actions)
        obs = self.positional_encoding(obs)
        actions = self.positional_encoding(actions)
        
        # concatenate observations and actions so that the sequence looks like this (s_0, a_0, s_1, a_1, ..., s_n, a_n)
        x = torch.stack((obs, actions), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_len, self.d_model)
        x = self.embedding_LN(x)
        x = self.dropout(x)

        # The Transformer Encoder with causal mask
        mask = torch.triu(torch.ones(2 * seq_len, 2 * seq_len) * float('-inf'), diagonal=1).cuda()
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_w = layer(x, mask=mask)
            attention_weights.append(attn_w)
        x = self.output(x)

        # split predictions into observations and actions predictions
        x = x.reshape(batch_size, seq_len, 2, self.action_dim).permute(0, 2, 1, 3)
        obs_pred, action_pred = x[:, 0], x[:, 1]

        return obs_pred, action_pred, attention_weights
