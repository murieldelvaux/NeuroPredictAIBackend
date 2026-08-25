from __future__ import annotations

import math
from typing import Tuple, Optional

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """
    Bloco Residual 3D padrão (BasicBlock 3D).
    
    Estrutura:
        Conv3D(3x3x3) -> BatchNorm3D -> GELU -> Conv3D(3x3x3) -> BatchNorm3D + Shortcut -> GELU
    
    Downsampling espacial é realizado via stride na primeira convolução 3D,
    evitando MaxPool3d (que apresenta incompatibilidades com MPS/Apple Silicon).
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.act2 = nn.GELU()

        # Shortcut de projeção 1x1x1 caso canais ou resolução espacial mudem
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class HybridCNNResNetTransformer3D(nn.Module):
    """
    Arquitetura Híbrida 3D: CNN Stem + ResNet Backbone + Vision Transformer + Grad-CAM.
    
    Projetada para classificação multi-classe de Alzheimer em volumes MRI T1 3D
    (CN: 0, MCI: 1, DEM: 2) a partir de tensores de entrada (B, 1, 128, 128, 128).

    Fluxo Arquitetural:
      1. Stem Convolucional 3D (128³ -> 64³): extrai features locais de baixo nível via Conv strided.
      2. Backbone Residual 3D em 3 estágios (64³ -> 32³ -> 16³ -> 8³):
         Gera um mapa volumétrico compacto (B, 256, 8, 8, 8).
      3. Tokenização 3D + Positional Encoding + CLS Token:
         Converte os 8x8x8 = 512 voxels em sequência de 512 tokens (+ 1 CLS token = 513 tokens).
      4. Transformer Encoder:
         Aplica blocos de Self-Attention Multi-Head (Pre-LayerNorm) para modelar
         correlações espaciais globais de longo alcance entre diferentes estruturas cerebrais.
      5. Head de Classificação:
         MLP sobre o token CLS com LayerNorm e Dropout para prever logits das 3 classes.
      6. Compatibilidade Grad-CAM 3D:
         Expõe a última camada convolucional `target_conv_layer` (nn.Conv3d) do backbone
         para registro de hooks forward/backward e geração de mapas de calor 3D interpretáveis.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int] = (64, 128, 256),
        embed_dim: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.4,
        spatial_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size

        # ── 1. Stem Convolucional 3D (128³ -> 64³) ────────────────────────────
        # Downsampling por stride=2 em vez de MaxPool3d (compatível com MPS Apple Silicon)
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.GELU(),
            nn.Conv3d(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.GELU(),
        )

        # ── 2. Backbone Residual 3D ───────────────────────────────────────────
        # Estágio 1: 64³ -> 32³ (stem_channels -> 64)
        self.stage1 = nn.Sequential(
            ResidualBlock3D(stem_channels, stage_channels[0], stride=2),
            ResidualBlock3D(stage_channels[0], stage_channels[0], stride=1),
        )

        # Estágio 2: 32³ -> 16³ (64 -> 128)
        self.stage2 = nn.Sequential(
            ResidualBlock3D(stage_channels[0], stage_channels[1], stride=2),
            ResidualBlock3D(stage_channels[1], stage_channels[1], stride=1),
        )

        # Estágio 3: 16³ -> 8³ (128 -> 256)
        self.stage3 = nn.Sequential(
            ResidualBlock3D(stage_channels[1], stage_channels[2], stride=2),
            ResidualBlock3D(stage_channels[2], stage_channels[2], stride=1),
        )

        # Camada alvo para Grad-CAM 3D: última Conv3d antes da tokenização do Transformer
        self.target_conv_layer: nn.Conv3d = self.stage3[-1].conv2

        # ── 3. Tokenização & Projeção Linear ──────────────────────────────────
        # Resolução espacial final: 128 / (2 * 2 * 2 * 2) = 8 -> 8³ = 512 tokens
        reduced_spatial = spatial_size // 16
        self.num_tokens = reduced_spatial ** 3

        if stage_channels[2] != embed_dim:
            self.token_proj = nn.Linear(stage_channels[2], embed_dim)
        else:
            self.token_proj = nn.Identity()

        # Token [CLS] global e Positional Embeddings 3D aprendíveis
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # ── 4. Transformer Encoder ───────────────────────────────────────────
        # Pre-LayerNorm (norm_first=True) confere maior estabilidade de convergência
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # ── 5. Head de Classificação ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização de pesos: trunc_normal_ para tokens e Kaiming para convoluções."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_gradcam_target_layer(self) -> nn.Conv3d:
        """
        Retorna a camada nn.Conv3d alvo para registro dos hooks do GradCAM3D.
        
        Corresponde à última camada convolucional do último bloco residual do Stage 3,
        preservando a topologia espacial 3D (8x8x8) antes da transformação em sequência.
        """
        return self.target_conv_layer

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai o mapa de features volumétrico do backbone CNN-ResNet: shape (B, C, D', H', W')."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

    def forward_transformer(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Tokeniza o mapa volumétrico 3D, adiciona positional embedding e processa no Transformer.
        Entrada: (B, C, D', H', W') -> Saída: (B, N+1, embed_dim)
        """
        b, c, d, h, w = feat.shape
        # Flatten espacial: (B, C, D*H*W) -> (B, N, C)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)

        # Concatena o [CLS] token no início da sequência: (B, N+1, embed_dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        seq = torch.cat((cls_tokens, tokens), dim=1)

        # Adiciona positional embedding 3D e dropout
        seq = seq + self.pos_embed
        seq = self.pos_drop(seq)

        # Passa pelo encoder de atenção global
        out = self.transformer_encoder(seq)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo.
        
        Args:
            x: Tensor do volume MRI de shape (B, 1, 128, 128, 128).
            
        Returns:
            Logits de shape (B, num_classes) [CN, MCI, DEM].
        """
        feat = self.extract_features(x)
        trans_out = self.forward_transformer(feat)

        # Vetor global extraído do [CLS] token (índice 0)
        cls_rep = trans_out[:, 0]
        logits = self.head(cls_rep)
        return logits
