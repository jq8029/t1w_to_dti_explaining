

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch
        self.norm1 = nn.BatchNorm3d(mid_ch)
        self.norm2 = nn.BatchNorm3d(mid_ch)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Identity()

    def forward(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x); x = self.norm2(x); x = self.relu(x)
        return x


class DownEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))


class UpDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2,
                          dy // 2, dy - dy // 2,
                          dz // 2, dz - dz // 2])
            
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutputConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * 4.0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        # x: (B, T, C)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BottleneckTransformer(nn.Module):
    """
    (B, C, D, H, W) -> tokens (B, T, E) -> transformer -> (B, C, D, H, W).
    """
    def __init__(self, in_ch):
        super().__init__()
        embed_dim = in_ch
        self.in_ch = in_ch
        self.embed_dim = embed_dim
    
        self.ln_in = nn.Linear(in_ch, embed_dim)
        self.ln_out = nn.Linear(embed_dim, in_ch)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim)
            for _ in range(4)
        ])

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # x: (b, c, d, h, w)
        B, C, D, H, W = x.shape
        
        T = D * H * W
        # reshape to (B, T, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, T, C)  # (B, T, C)
        # linear in channels -> embed
        x = self.ln_in(x)  # (B, T, E)

        # register a new parameter sized (1, T, E)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, self.embed_dim, device=x.device))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        x = x + self.pos_emb
        x = self.dropout(x)

        # transformer layers
        for layer in self.layers:
            x = layer(x)  # (B, T, E)
        # project back to channel dim
        x = self.ln_out(x)  # (B, T, C)
        # reshape back to (B, C, D, H, W)
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        return x

class OutConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base=32, two_head=False):
        super().__init__()
        self.two_head = two_head

        # Encoder
        self.inc   = DoubleConv(in_channels,     base)
        self.down1 = DownEncoder(base,           base * 2)
        self.down2 = DownEncoder(base * 2,       base * 4)
        self.down3 = DownEncoder(base * 4,       base * 8)

        # Bottleneck CNN
        self.bot_conv = DoubleConv(base * 8, base * 16)
        # Bottleneck Transformer
        self.bot_trans = BottleneckTransformer(in_ch=base*16)

        # Decoder
        self.up3   = UpDecoder(base * 16,        base * 8)
        self.up2   = UpDecoder(base * 8,         base * 4)
        self.up1   = UpDecoder(base * 4,         base * 2)
        self.up0   = UpDecoder(base * 2,         base)

        if (self.two_head):
            # FA head
            self.fa_head  = OutConv3d(base, 1)
            # MD + AD + RD head
            self.dti_head = OutConv3d(base, 3)
        else:
            self.outc  = OutConv3d(base,        out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):   
        # x: [N, C, D, H, W]
        c0 = self.inc(x)        # 32
        c1 = self.down1(c0)     # 64
        c2 = self.down2(c1)     # 128
        c3 = self.down3(c2)     # 256
        b  = self.bot_conv(c3)  # 512
        b = self.bot_trans(b)   # 512
        u3 = self.up3(b,  c3)   # 256
        u2 = self.up2(u3, c2)   # 128
        u1 = self.up1(u2, c1)   # 64
        u0 = self.up0(u1, c0)   # 32

        if (self.two_head):
            fa  = self.fa_head(u0)
            dti = self.dti_head(u0)
            return fa, dti
        else:
            y = self.outc(u0)
            return y


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base=32):
        super().__init__()
        self.two_head = False

        # Encoder
        self.inc   = DoubleConv(in_channels,     base)
        self.down1 = DownEncoder(base,           base * 2)
        self.down2 = DownEncoder(base * 2,       base * 4)
        self.down3 = DownEncoder(base * 4,       base * 8)

        # Bottleneck
        self.bot   = DoubleConv(base * 8,        base * 16)

        # Decoder
        self.up3   = UpDecoder(base * 16,        base * 8)
        self.up2   = UpDecoder(base * 8,         base * 4)
        self.up1   = UpDecoder(base * 4,         base * 2)
        self.up0   = UpDecoder(base * 2,         base)

        self.outc  = OutConv3d(base,             out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [N, C, D, H, W]
        c0 = self.inc(x)        # 32
        c1 = self.down1(c0)     # 64
        c2 = self.down2(c1)     # 128
        c3 = self.down3(c2)     # 256
        b  = self.bot(c3)       # 512
        u3 = self.up3(b,  c3)   # 256
        u2 = self.up2(u3, c2)   # 128
        u1 = self.up1(u2, c1)   # 64
        u0 = self.up0(u1, c0)   # 32
        y  = self.outc(u0)      # 4
        return y


class Up3d_no_skip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    
class CNN3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base=32):
        super().__init__()
        self.two_head = False
        
        # Encoder
        self.inc   = DoubleConv(in_channels,     base)
        self.down1 = DownEncoder(base,           base * 2)
        self.down2 = DownEncoder(base * 2,       base * 4)
        self.down3 = DownEncoder(base * 4,       base * 8)

        # Bottleneck
        self.bot = DoubleConv(base * 8, base * 16)

        # Decoder (NO skip connections)
        self.up3 = Up3d_no_skip(base * 16, base * 8)
        self.up2 = Up3d_no_skip(base * 8, base * 4)
        self.up1 = Up3d_no_skip(base * 4, base * 2)
        self.up0 = Up3d_no_skip(base * 2, base)
        self.reduce = nn.Conv3d(base * 2, base, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv3d(base, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, out_channels, kernel_size=1)   
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [N, C, D, H, W]

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        b = self.bot(x4)

        # Decoder (no skip)
        u3 = self.up3(b)
        u2 = self.up2(u3)
        u1 = self.up1(u2)
        u0 = self.reduce(u1)
        out = self.head(u0)

        return out
