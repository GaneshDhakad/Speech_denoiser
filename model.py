import torch
import torch.nn as nn

class CNNDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(CNNDenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input shape: (B, 1, 129, 251)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 65, 126)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 33, 63)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 17, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, 9, 16)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)), # (B, 64, 17, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)), # (B, 32, 33, 63)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)), # (B, 16, 65, 126)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)), # (B, 1, 129, 251)
            nn.Sigmoid() # Mask values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        mask = self.decoder(encoded)
        
        # Ensure mask precisely matches the size of x due to odd dimensions
        if mask.shape != x.shape:
            mask = torch.nn.functional.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            
        denoised = x * mask
        return denoised, mask

