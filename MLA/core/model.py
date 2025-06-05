import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rotary_matrix(seq_len, d_model, device, theta_base=10000.0, eps=1e-6):
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    theta = 1.0 / (theta_base ** (2.0 * torch.arange(0, d_model, 2, device=device).float() / d_model + eps))
    positions = torch.arange(seq_len, device=device).float().unsqueeze(1)
    angles = positions * theta
    angles = torch.clamp(angles, min=-1e4, max=1e4)
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    rotary_matrix = torch.stack([cosines, -sines, sines, cosines], dim=-1)
    rotary_matrix = rotary_matrix.view(seq_len, d_model // 2, 2, 2)
    return rotary_matrix

def apply_rotary_embeddings(x, rotary_matrix):
    batch_size, seq_len, d_model = x.shape
    rot_seq_len, rot_dim, _, _ = rotary_matrix.shape
    if rot_seq_len < seq_len:
        raise ValueError(f"Rotary matrix seq_len {rot_seq_len} is too short for input seq_len {seq_len}")
    if rot_dim != d_model // 2:
        raise ValueError(f"Rotary matrix dim {rot_dim} does not match d_model // 2 = {d_model // 2}")
    x_reshaped = x.view(batch_size, seq_len, d_model // 2, 2)
    rotary_matrix = rotary_matrix[:seq_len].unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    x_rotated = torch.einsum('bsdh,bsdij->bsdh', x_reshaped, rotary_matrix)
    return x_rotated.view(batch_size, seq_len, d_model)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, d_c, d_prime_c, d_R_h):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_h = d_model // nhead
        self.d_c = d_c
        self.d_prime_c = d_prime_c
        self.d_R_h = d_R_h

        # Projection matrices for MLA
        self.W_DKV = nn.Linear(d_model, self.d_c, bias=False)
        self.W_UK = nn.Linear(self.d_c, d_model, bias=False)
        self.W_UV = nn.Linear(self.d_c, d_model, bias=False)
        self.W_KR = nn.Linear(d_model, self.d_R_h, bias=False)
        self.W_DQ = nn.Linear(d_model, self.d_prime_c, bias=False)
        self.W_UQ = nn.Linear(self.d_prime_c, d_model, bias=False)
        self.W_QR = nn.Linear(self.d_prime_c, self.d_R_h * nhead, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.W_DKV.weight)
        nn.init.xavier_uniform_(self.W_UK.weight)
        nn.init.xavier_uniform_(self.W_UV.weight)
        nn.init.xavier_uniform_(self.W_KR.weight)
        nn.init.xavier_uniform_(self.W_DQ.weight)
        nn.init.xavier_uniform_(self.W_UQ.weight)
        nn.init.xavier_uniform_(self.W_QR.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        batch_size, seq_len, _ = tgt.shape
        device = tgt.device

        # Compute compressed KV latent vector
        c_KV = self.W_DKV(tgt)
        k_C = self.W_UK(c_KV).view(batch_size, seq_len, self.nhead, self.d_h)
        v_C = self.W_UV(c_KV).view(batch_size, seq_len, self.nhead, self.d_h)

        # Compute decoupled key with RoPE
        temp_k = self.W_KR(tgt)
        rotary_matrix = get_rotary_matrix(seq_len, self.d_R_h, device)
        k_R = apply_rotary_embeddings(temp_k, rotary_matrix)

        # Compute compressed query latent vector
        c_Q = self.W_DQ(tgt)
        q_C = self.W_UQ(c_Q).view(batch_size, seq_len, self.nhead, self.d_h)

        # Compute decoupled queries with RoPE
        temp_q = self.W_QR(c_Q).view(batch_size, seq_len, self.nhead, self.d_R_h)
        temp_q_reshaped = temp_q.permute(0, 2, 1, 3).reshape(batch_size * self.nhead, seq_len, self.d_R_h)
        q_R_reshaped = apply_rotary_embeddings(temp_q_reshaped, rotary_matrix)
        q_R = q_R_reshaped.view(batch_size, self.nhead, seq_len, self.d_R_h).permute(0, 2, 1, 3)

        # Form final queries, keys, and values
        Q = torch.cat([q_C, q_R], dim=-1).permute(0, 2, 1, 3)  # [batch_size, nhead, seq_len, d_h + d_R_h]
        K = torch.cat([k_C, k_R.unsqueeze(2).expand(-1, -1, self.nhead, -1)], dim=-1).permute(0, 2, 1, 3)
        V = v_C.permute(0, 2, 1, 3)  # [batch_size, nhead, seq_len, d_h]

        # Prepare attention mask
        attn_mask = None
        if tgt_mask is not None:
            # Expect tgt_mask to be [batch_size * nhead, seq_len, seq_len]
            attn_mask = tgt_mask.view(batch_size, self.nhead, seq_len, seq_len).to(tgt.dtype)  # [batch_size, nhead, seq_len, seq_len]
        if tgt_key_padding_mask is not None:
            # Convert padding mask to [batch_size, nhead, seq_len]
            padding_mask = tgt_key_padding_mask.unsqueeze(1).expand(-1, self.nhead, -1)  # [batch_size, nhead, seq_len]
            # Create a broadcastable mask for attention
            padding_mask = padding_mask.unsqueeze(-2) * -1e9  # [batch_size, nhead, 1, seq_len]
            if attn_mask is None:
                attn_mask = padding_mask
            else:
                attn_mask = attn_mask + padding_mask

        # Compute attention output
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0
        )
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.W_O(attn_output)

        # Residual connection and normalization
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(ff_output))
        return tgt

class StoryTellerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, embed_dropout, num_genres, d_c, d_prime_c, d_R_h):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for RoPE, got {d_model}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if d_R_h % 2 != 0:
            raise ValueError(f"d_R_h must be even for RoPE, got {d_R_h}")
        self.nhead = nhead
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout, d_c, d_prime_c, d_R_h) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.genre_head = nn.Linear(d_model, num_genres)
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.xavier_uniform_(self.genre_head.weight)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        batch_size, seq_len = tgt.shape
        device = tgt.device
        x = self.embedding(tgt)
        x = self.embed_dropout(x)
        for layer in self.decoder_layers:
            x = layer(tgt=x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        x = self.norm(x)
        lm_logits = self.lm_head(x)
        return lm_logits