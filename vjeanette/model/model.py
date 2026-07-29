import torch.nn as nn
import torch

import torch.nn as nn

class UNet1D_Embed(nn.Module):
    def __init__(self, embed_dim=32, out_channels=1, features=[8,16,64,64]):
        super().__init__()
        PADDING_IDX = 5
        self.embed = nn.Embedding(num_embeddings=6, embedding_dim=embed_dim, padding_idx=PADDING_IDX)
        
        in_channels = embed_dim
        
        # Encoder
        self.enc1 = self._encoder_block(in_channels, features[0])
        self.enc2 = self._encoder_block(features[0], features[1])
        self.enc3 = self._encoder_block(features[1], features[2])
        self.enc4 = self._encoder_block(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(features[3], features[3]*2, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(features[3]*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(features[3]*2, features[3]*2, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(features[3]*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose1d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(features[3]+features[2], features[2])
        
        self.up3 = nn.ConvTranspose1d(features[2], features[2], kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(features[2]+features[1], features[1])
        
        self.up2 = nn.ConvTranspose1d(features[1], features[1], kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(features[1]+features[0], features[0])
        
        self.up1 = nn.ConvTranspose1d(features[0], features[0], kernel_size=2, stride=2)
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(features[0], features[0], kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(features[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(features[0], features[0], kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(features[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.v_head = nn.Sequential(
            nn.Conv1d(features[0], features[0]//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features[0]//2, out_channels, kernel_size=1)
        )
        self.j_head = nn.Sequential(
            nn.Conv1d(features[0], features[0]//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features[0]//2, out_channels, kernel_size=1)
        )
        self.cdr_head = nn.Sequential(
            nn.Conv1d(features[0], features[0]//2, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv1d(features[0]//2, out_channels, kernel_size = 1)
        )
        
    
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.MaxPool1d(2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35)
        )
    
    def forward(self, x):
        """
        x: [B, L] LongTensor индексы 0..5
        """
        x = self.embed(x).transpose(1, 2)  # [B, embed_dim, L]
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.up4(b)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self.final_conv(d1)
        
        v_out = self.v_head(d1)
        j_out = self.j_head(d1)
        cdr_out = self.cdr_head(d1)
        
        return cdr_out.squeeze(1), v_out.squeeze(1), j_out.squeeze(1)