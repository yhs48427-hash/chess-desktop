"""Hierarchical Chess WDL 모델. (Drive chess/chesswdl 의 추론용 사본)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import (ELO_BUCKETS, HALFMOVE_MAX, N_CONT, NUM_CASTLE, NUM_EP, NUM_PIECE_TOKENS,
                       PLY_BUCKETS, POLICY_SIZE)

N_EXTRA = 5
N_TOKENS = 64 + N_EXTRA

DEFAULT_CFG = dict(
    d_model=256, n_heads=8, spatial_layers=8, ffn_mult=4, dropout=0.1,
    temporal="gru", temporal_layers=2, window=32, player_dim=64,
)


class Attention(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        assert d % h == 0
        self.h, self.dropout = h, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, bias=None, causal=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if bias is not None:
            bias = bias.to(q.dtype).unsqueeze(0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=causal and bias is None)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h, ffn_mult, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, h, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias=None, causal=False):
        x = x + self.drop(self.attn(self.ln1(x), bias, causal))
        return x + self.drop(self.ffn(self.ln2(x)))


class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["d_model"], cfg["n_heads"]
        self.piece_emb = nn.Embedding(NUM_PIECE_TOKENS, d)
        self.sq_pos = nn.Parameter(torch.randn(64, d) * 0.02)
        self.castle_emb = nn.Embedding(NUM_CASTLE, d)
        self.ep_emb = nn.Embedding(NUM_EP, d)
        self.hm_emb = nn.Embedding(HALFMOVE_MAX + 1, d)
        self.ply_emb = nn.Embedding(PLY_BUCKETS, d)
        self.elo_emb = nn.Embedding(ELO_BUCKETS, d)
        self.cont_proj = nn.Sequential(nn.Linear(N_CONT, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.randn(N_EXTRA, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h, cfg["ffn_mult"], cfg["dropout"])
                                     for _ in range(cfg["spatial_layers"])])
        self.attn_bias = nn.Parameter(torch.zeros(cfg["spatial_layers"], h, N_TOKENS, N_TOKENS))
        self.ln = nn.LayerNorm(d)
        self.grad_ckpt = False

    def forward(self, boards, state, elo, cont):
        sq = self.piece_emb(boards) + self.sq_pos
        st = (self.castle_emb(state[:, 0]) + self.ep_emb(state[:, 1])
              + self.hm_emb(state[:, 2]) + self.ply_emb(state[:, 3]))
        extra = torch.stack([st, self.elo_emb(elo[:, 0]), self.elo_emb(elo[:, 1]),
                             self.cont_proj(cont), torch.zeros_like(st)], 1) + self.type_emb
        x = torch.cat([sq, extra], 1)
        for i, blk in enumerate(self.blocks):
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, self.attn_bias[i], use_reentrant=False)
            else:
                x = blk(x, self.attn_bias[i])
        return self.ln(x[:, -1])


class TemporalGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.gru = nn.GRU(d, d, num_layers=cfg["temporal_layers"], batch_first=True,
                          dropout=cfg["dropout"] if cfg["temporal_layers"] > 1 else 0.0)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            y, _ = self.gru(x.float())
        return y


class TemporalTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.pos = nn.Parameter(torch.randn(cfg["window"], d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["n_heads"], cfg["ffn_mult"], cfg["dropout"])
                                     for _ in range(cfg["temporal_layers"])])

    def forward(self, x):
        x = x + self.pos[: x.shape[1]]
        for blk in self.blocks:
            x = blk(x, causal=True)
        return x


class ChessWDLModel(nn.Module):
    def __init__(self, cfg: dict, num_players: int):
        super().__init__()
        self.cfg = {**DEFAULT_CFG, **cfg}
        cfg = self.cfg
        d, p = cfg["d_model"], cfg["player_dim"]
        self.spatial = SpatialEncoder(cfg)
        self.temporal = TemporalGRU(cfg) if cfg["temporal"] == "gru" else TemporalTransformer(cfg)
        self.player_emb = nn.Embedding(num_players, p)
        nn.init.normal_(self.player_emb.weight, std=0.02)
        with torch.no_grad():
            self.player_emb.weight[0].zero_()
        self.cond_in = nn.Linear(2 * p, d)
        self.film = nn.Linear(2 * p, 2 * d)
        for m in (self.cond_in, self.film):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.ln_out = nn.LayerNorm(d)
        self.wdl_head = nn.Linear(d, 3)
        self.policy_head = nn.Linear(d, POLICY_SIZE)

    @property
    def num_players(self):
        return self.player_emb.num_embeddings

    def encode_positions(self, batch):
        mask = batch["mask"]
        h_flat = self.spatial(batch["boards"][mask], batch["state"][mask],
                              batch["elo"][mask], batch["cont"][mask])
        h = h_flat.new_zeros(mask.shape + (h_flat.shape[-1],))
        h[mask] = h_flat
        return h

    def temporal_features(self, h, players):
        c = self.player_emb(players).flatten(-2)
        x = self.temporal(h + self.cond_in(c))
        gamma, beta = self.film(c).chunk(2, -1)
        return self.ln_out(x * (1 + gamma) + beta)

    def temporal_heads(self, h, players):
        z = self.temporal_features(h, players)
        return self.wdl_head(z), self.policy_head(z)

    def last_features(self, h, players, lengths):
        z = self.temporal_features(h, players)
        last = (lengths - 1).clamp(min=0)
        return z[torch.arange(z.shape[0], device=z.device), last]

    def heads_at(self, h, players, start, end, valid):
        B, S, d = h.shape
        M = start.shape[1]
        lens = ((end - start + 1).clamp(min=1) * valid).long()
        L = max(1, int(lens.max()))
        ar = torch.arange(L, device=h.device)
        ok = ar.view(1, 1, L) < lens.unsqueeze(-1)
        idx = torch.where(ok, start.clamp(min=0).unsqueeze(-1) + ar, torch.zeros_like(ok, dtype=torch.long))
        idx = idx.clamp(max=S - 1).view(B, M * L, 1)
        hh = torch.gather(h, 1, idx.expand(-1, -1, d)).view(B * M, L, d)
        hh = hh * ok.view(B * M, L, 1).to(hh.dtype)
        pp = torch.gather(players, 1, idx.expand(-1, -1, 2)).view(B * M, L, 2)
        zl = self.last_features(hh, pp, lens.view(-1))
        return self.wdl_head(zl).view(B, M, 3), self.policy_head(zl).view(B, M, -1)

    def forward(self, batch):
        h = self.encode_positions(batch)
        return self.heads_at(h, batch["players"], batch["t_start"], batch["t_end"], batch["t_mask"])
