import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder2D(nn.Module):
    def __init__(self, in_channels=1, base_channels=24, out_dim=32, dropout=0.2):
        super().__init__()
        mid_channels = max(base_channels, out_dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_dim)

    def forward(self, x):
        # x: (B,T,1,H,W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.net(x).reshape(b * t, self.out_dim)
        x = self.dropout(x)
        return x.reshape(b, t, self.out_dim)


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 2, 4), dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2) * int(d),
                        dilation=int(d),
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )
                for d in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        branch_outputs = [branch(x) for branch in self.branches]
        fused = self.fuse(torch.cat(branch_outputs, dim=1))
        fused = self.dropout(fused)
        return F.relu(fused + residual, inplace=True)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.attn = nn.Linear(channels, 1)
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = int(channels)

    def forward(self, x):
        # x: (B,T,C)
        weights = torch.softmax(self.attn(x), dim=1)
        attn_feat = torch.sum(weights * x, dim=1)
        mean_feat = torch.mean(x, dim=1)
        max_feat = torch.max(x, dim=1).values
        pooled = torch.cat([attn_feat, mean_feat, max_feat], dim=1)
        return self.proj(pooled), weights


class TemporalSequencePooling(nn.Module):
    def __init__(self, channels, mode="attention", dropout=0.2):
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"attention", "mean", "max", "center", "last"}:
            raise ValueError(f"Unsupported temporal pooling mode: {mode}")
        self.out_dim = int(channels)
        if self.mode == "attention":
            self.pool = TemporalAttentionPooling(channels, dropout=dropout)
        else:
            self.pool = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        # x: (B,T,C)
        if self.mode == "attention":
            return self.pool(x)
        b, t, _c = x.shape
        if self.mode == "mean":
            feat = torch.mean(x, dim=1)
            weights = torch.full((b, t, 1), 1.0 / max(t, 1), dtype=x.dtype, device=x.device)
        elif self.mode == "max":
            feat = torch.max(x, dim=1).values
            max_idx = torch.argmax(torch.norm(x, p=2, dim=2), dim=1)
            weights = torch.zeros((b, t, 1), dtype=x.dtype, device=x.device)
            weights.scatter_(1, max_idx.view(b, 1, 1), 1.0)
        elif self.mode == "center":
            center_idx = t // 2
            feat = x[:, center_idx, :]
            weights = torch.zeros((b, t, 1), dtype=x.dtype, device=x.device)
            weights[:, center_idx, 0] = 1.0
        else:
            feat = x[:, -1, :]
            weights = torch.zeros((b, t, 1), dtype=x.dtype, device=x.device)
            weights[:, -1, 0] = 1.0
        return self.pool(feat), weights


class WindowAttentionPooling(nn.Module):
    def __init__(self, channels, dropout=0.2, center_prior_scale_init=0.75):
        super().__init__()
        hidden = max(16, channels // 2)
        self.attn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.center_logit_scale = nn.Parameter(torch.tensor(float(center_prior_scale_init), dtype=torch.float32))
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = int(channels)

    def forward(self, x):
        # x: (B,K,C)
        b, k, _c = x.shape
        logits = self.attn(x)
        if k > 1:
            pos = torch.linspace(-1.0, 1.0, steps=k, device=x.device, dtype=x.dtype).view(1, k, 1)
            center_prior = -(pos ** 2)
            logits = logits + F.softplus(self.center_logit_scale) * center_prior
        weights = torch.softmax(logits, dim=1)
        attn_feat = torch.sum(weights * x, dim=1)
        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        pooled = torch.cat([attn_feat, mean_feat, max_feat], dim=1)
        return self.proj(pooled), weights


class DualStreamMSTCNDetector(nn.Module):
    def __init__(
        self,
        seq_len=10,
        frame_feature_dim=32,
        temporal_channels=64,
        temporal_blocks=3,
        dropout=0.35,
        use_delta_branch=True,
        temporal_pooling="attention",
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.use_delta_branch = bool(use_delta_branch)
        self.temporal_pooling = str(temporal_pooling).strip().lower()

        self.raw_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.25, dropout * 0.5),
        )
        self.delta_encoder = (
            FrameEncoder2D(
                in_channels=1,
                base_channels=max(16, frame_feature_dim),
                out_dim=frame_feature_dim,
                dropout=min(0.25, dropout * 0.5),
            )
            if self.use_delta_branch
            else None
        )

        temporal_input_dim = frame_feature_dim * (2 if self.use_delta_branch else 1)
        self.temporal_input = nn.Sequential(
            nn.Conv1d(temporal_input_dim, temporal_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                MultiScaleTemporalBlock(
                    channels=temporal_channels,
                    kernel_size=3,
                    dilations=(1, 2, 4),
                    dropout=min(0.35, dropout),
                )
                for _ in range(int(temporal_blocks))
            ]
        )
        self.pooling = TemporalSequencePooling(
            temporal_channels,
            mode=self.temporal_pooling,
            dropout=min(0.35, dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling.out_dim, max(32, temporal_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(32, temporal_channels // 2), 1),
        )
        self.feature_dim = int(self.pooling.out_dim)

    @staticmethod
    def compute_delta(x):
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    def encode_sequence(self, x):
        raw_seq = self.raw_encoder(x)
        streams = [raw_seq]
        delta_seq = None
        if self.use_delta_branch:
            delta_x = self.compute_delta(x)
            delta_seq = self.delta_encoder(delta_x)
            streams.append(delta_seq)

        seq = torch.cat(streams, dim=-1)  # (B,T,C)
        seq = seq.transpose(1, 2)  # (B,C,T)
        seq = self.temporal_input(seq)
        for block in self.temporal_blocks:
            seq = block(seq)
        temporal_seq = seq.transpose(1, 2)  # (B,T,C)
        features, attn_weights = self.pooling(temporal_seq)
        return {
            "raw_seq": raw_seq,
            "delta_seq": delta_seq,
            "temporal_seq": temporal_seq,
            "pooled_features": features,
            "attn_weights": attn_weights,
        }

    def forward(self, x, return_features=False):
        feats = self.encode_sequence(x)
        logit = self.classifier(feats["pooled_features"])
        if return_features:
            return logit, feats
        return logit


class DualStreamMSTCNContextDetector(nn.Module):
    def __init__(
        self,
        seq_len=10,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.35,
        frame_feature_dim=48,
        temporal_channels=96,
        temporal_blocks=3,
        use_delta_branch=False,
        context_heads=2,
        context_layers=1,
        max_context_windows=9,
        temporal_pooling="attention",
        **_kwargs,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.max_context_windows = int(max_context_windows)
        self.window_encoder = DualStreamMSTCNDetector(
            seq_len=seq_len,
            frame_feature_dim=frame_feature_dim,
            temporal_channels=temporal_channels,
            temporal_blocks=temporal_blocks,
            dropout=dropout,
            use_delta_branch=use_delta_branch,
            temporal_pooling=temporal_pooling,
        )
        ctx_dim = int(self.window_encoder.feature_dim)
        self.window_pos_embed = nn.Parameter(torch.zeros(1, self.max_context_windows, ctx_dim))
        nn.init.trunc_normal_(self.window_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim,
            nhead=int(context_heads),
            dim_feedforward=max(ctx_dim * 2, 128),
            dropout=min(0.20, dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(context_layers))
        self.context_pool = WindowAttentionPooling(ctx_dim, dropout=min(0.25, dropout))
        self.classifier = nn.Sequential(
            nn.Linear(self.context_pool.out_dim, max(48, ctx_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(48, ctx_dim // 2), 1),
        )
        self.feature_dim = int(self.context_pool.out_dim)

    def forward(self, x, return_features=False):
        # Supports either (B,T,1,H,W) or (B,K,T,1,H,W)
        if x.dim() == 5:
            x = x.unsqueeze(1)
        if x.dim() != 6:
            raise ValueError(f"Expected input dims 5 or 6, got shape {tuple(x.shape)}")

        b, k, t, c, h, w = x.shape
        if k > self.max_context_windows:
            raise ValueError(f"context windows {k} exceed max_context_windows={self.max_context_windows}")

        flat_x = x.reshape(b * k, t, c, h, w)
        window_feats = self.window_encoder.encode_sequence(flat_x)
        window_tokens = window_feats["pooled_features"].reshape(b, k, -1)
        window_attn = window_feats["attn_weights"].reshape(b, k, -1)

        ctx_tokens = window_tokens + self.window_pos_embed[:, :k]
        ctx_tokens = self.context_encoder(ctx_tokens)
        pooled, context_weights = self.context_pool(ctx_tokens)
        logit = self.classifier(pooled)

        if return_features:
            return logit, {
                "window_tokens": window_tokens,
                "context_tokens": ctx_tokens,
                "window_attention": context_weights,
                "frame_attention": window_attn,
                "pooled_features": pooled,
            }
        return logit


class DualStreamMSTCNContextResidualDetector(nn.Module):
    def __init__(
        self,
        seq_len=10,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.35,
        frame_feature_dim=48,
        temporal_channels=96,
        temporal_blocks=3,
        use_delta_branch=False,
        context_heads=2,
        max_context_windows=7,
        context_hidden_dim=96,
        temporal_pooling="attention",
        **_kwargs,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.max_context_windows = int(max_context_windows)
        self.window_encoder = DualStreamMSTCNDetector(
            seq_len=seq_len,
            frame_feature_dim=frame_feature_dim,
            temporal_channels=temporal_channels,
            temporal_blocks=temporal_blocks,
            dropout=dropout,
            use_delta_branch=use_delta_branch,
            temporal_pooling=temporal_pooling,
        )
        ctx_dim = int(self.window_encoder.feature_dim)
        self.window_pos_embed = nn.Parameter(torch.zeros(1, self.max_context_windows, ctx_dim))
        nn.init.trunc_normal_(self.window_pos_embed, std=0.02)
        self.context_norm = nn.LayerNorm(ctx_dim)
        self.context_attn = nn.MultiheadAttention(
            embed_dim=ctx_dim,
            num_heads=int(context_heads),
            dropout=min(0.15, dropout),
            batch_first=True,
        )
        self.local_classifier = nn.Sequential(
            nn.Linear(ctx_dim, max(48, ctx_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(48, ctx_dim // 2), 1),
        )
        self.context_delta_head = nn.Sequential(
            nn.Linear(ctx_dim * 3, int(max(64, context_hidden_dim))),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(max(64, context_hidden_dim)), 1),
        )
        self.context_scale_raw = nn.Parameter(torch.tensor(-1.6, dtype=torch.float32))
        self.feature_dim = int(ctx_dim)

    def forward(self, x, return_features=False):
        if x.dim() == 5:
            x = x.unsqueeze(1)
        if x.dim() != 6:
            raise ValueError(f"Expected input dims 5 or 6, got shape {tuple(x.shape)}")

        b, k, t, c, h, w = x.shape
        if k > self.max_context_windows:
            raise ValueError(f"context windows {k} exceed max_context_windows={self.max_context_windows}")

        flat_x = x.reshape(b * k, t, c, h, w)
        window_feats = self.window_encoder.encode_sequence(flat_x)
        window_tokens = window_feats["pooled_features"].reshape(b, k, -1)
        frame_attention = window_feats["attn_weights"].reshape(b, k, -1)

        center_idx = k // 2
        center_token = window_tokens[:, center_idx : center_idx + 1, :]
        center_pos = self.window_pos_embed[:, center_idx : center_idx + 1]
        tokens_with_pos = self.context_norm(window_tokens + self.window_pos_embed[:, :k])
        context_token, context_weights = self.context_attn(
            query=self.context_norm(center_token + center_pos),
            key=tokens_with_pos,
            value=tokens_with_pos,
            need_weights=True,
        )

        center_token_flat = center_token.squeeze(1)
        context_token_flat = context_token.squeeze(1)
        local_logit = self.local_classifier(center_token_flat)
        delta_input = torch.cat(
            [
                center_token_flat,
                context_token_flat,
                context_token_flat - center_token_flat,
            ],
            dim=1,
        )
        delta_logit = self.context_delta_head(delta_input)
        context_scale = 0.35 * torch.sigmoid(self.context_scale_raw)
        logit = local_logit + context_scale * delta_logit

        if return_features:
            return logit, {
                "window_tokens": window_tokens,
                "center_token": center_token_flat,
                "context_token": context_token_flat,
                "local_logit": local_logit,
                "delta_logit": delta_logit,
                "context_scale": context_scale.detach(),
                "window_attention": context_weights,
                "frame_attention": frame_attention,
            }
        return logit
