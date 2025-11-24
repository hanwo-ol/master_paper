import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, N, T = x.size()
        proj_query = self.query(x).view(B, -1, N * T).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, N * T)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(B, -1, N * T)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, N, T)
        
        return self.gamma * out + x

class AssetUNet(nn.Module):
    def __init__(self, n_features=38, n_assets=501):
        super().__init__()
        
        # [추가] 입력 데이터 안정화를 위한 Instance Normalization
        # (Batch, Feature, Asset, Time)에서 Feature별로 정규화
        self.input_norm = nn.InstanceNorm2d(n_features, affine=False)
        
        # Encoder
        self.enc1 = ConvBlock(n_features, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 256)
        self.attention = SelfAttentionBlock(256)
        
        # Decoder
        self.dec3 = ConvBlock(256 + 256, 128) 
        self.dec2 = ConvBlock(128 + 128, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        # [추가] 가중치 초기화 (Kaiming He)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, F, N, T)
        
        # [추가] 입력 정규화 적용
        # 값이 너무 크거나 작으면 여기서 잡힘
        x = self.input_norm(x)
        
        # Encoder
        e1 = self.enc1(x)          # (B, 64, N, T)
        p1 = self.pool(e1)         # (B, 64, N, T/2)
        
        e2 = self.enc2(p1)         # (B, 128, N, T/2)
        p2 = self.pool(e2)         # (B, 128, N, T/4)
        
        e3 = self.enc3(p2)         # (B, 256, N, T/4)
        p3 = self.pool(e3)         # (B, 256, N, T/8)
        
        # Bottleneck
        b = self.bottleneck(p3)
        b = self.attention(b)      # (B, 256, N, T/8)
        
        # Decoder
        d3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1) 
        d3 = self.dec3(d3)              
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1) 
        d2 = self.dec2(d2)              
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1) 
        d1 = self.dec1(d1)              
        
        out = self.final(d1) # (B, 1, N, T)
        
        pred = out[..., -1].squeeze(-1) # (B, 1, N)
        return pred.squeeze(1) # (B, N)

if __name__ == "__main__":
    # Test Model
    x = torch.randn(2, 38, 501, 60)
    model = AssetUNet(n_features=38, n_assets=501)
    y = model(x)
    print("\n[Model Output Check]")
    print("Input:", x.shape)
    print("Output:", y.shape)