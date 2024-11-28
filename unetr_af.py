import torch
import torch.nn as nn
import timm
from model import attention_gate
from fvcore.nn import FlopCountAnalysis
from transformers import ViTModel, EncoderDecoderModel
from torchvision import models
from cbam import *
from se_block import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.deconv(x)


class UNetR2D_AF(nn.Module):
    def __init__(self, cg, vit_model="google/vit-base-patch16-224-in21k", pretrained=True, num_classes=1):
        super().__init__()
        self.cg = cg
        
        # Load pre-trained Vision Transformer (ViT) model
        self.vit = ViTModel.from_pretrained(vit_model, output_hidden_states=True)
        
        # CNN Decoder
        self.d1 = DeconvBlock(cg["hidden_dim"], 512)
        self.s1 = nn.Sequential(
            DeconvBlock(cg["hidden_dim"], 512),
            ConvBlock(512, 512),
            SELayer(512, reduction=16),
        )
        self.c1 = nn.Sequential(
            ConvBlock(512 + 512, 512),
            ConvBlock(512, 512),
            CBAM(512),
        )

        # Decoder 2
        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(cg["hidden_dim"], 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256),
            SELayer(256, reduction=16),
        )
        self.c2 = nn.Sequential(
            ConvBlock(256 + 256, 256),
            ConvBlock(256, 256),
            CBAM(256),
        )
    
        # Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(cg["hidden_dim"], 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            SELayer(128, reduction=16),
        )
        self.c3 = nn.Sequential(
            ConvBlock(128 + 128, 128),
            ConvBlock(128, 128),
            CBAM(128),
        )

        # Decoder 4
        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            SELayer(64, reduction=16),
        )
        self.c4 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ConvBlock(64, 64),
            CBAM(64),
        )

        # Final Layer
        self.output = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Get hidden states from ViT model
        vit_outputs = self.vit(pixel_values=x)
        hidden_states = vit_outputs.hidden_states  # Tuple of hidden states at each layer

        # Reshape the hidden states from (batch_size, sequence_length, hidden_dim) to (batch_size, hidden_dim, H, W)
        # The sequence_length should be num_patches + 1 (for the class token), so the class token is excluded and the patches are reshaped
        batch_size, seq_length, hidden_dim = hidden_states[-1].size()
        
        patch_size = self.cg["patch_size"]
        grid_size = int((seq_length - 1) ** 0.5)  # For 197 tokens, grid_size would be 14 for 14x14 patches

        # Remove the class token and reshape
        z12 = hidden_states[-1][:, 1:, :].reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)

        # Decoder 1
        x = self.d1(z12)
        x = nn.Dropout(0.1)(x)
        z9 = hidden_states[9][:, 1:, :].reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # Decoder 2
        x = self.d2(x)
        x = nn.Dropout(0.1)(x)
        z6 = hidden_states[6][:, 1:, :].reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        # Decoder 3
        x = self.d3(x)
        z3 = hidden_states[3][:, 1:, :].reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        # Decoder 4
        x = self.d4(x)
        s = self.s4(x)  # Assuming the original input image is used for the final skip connection
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        # Final Layer
        output = self.output(x)

        return output


# Configuration settings
if __name__ == '__main__':
    config = {}
    config["image_size"] = 224  # Set to 224
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = (config["image_size"] // 16) ** 2  # Since 224 / 16 = 14
    config["patch_size"] = 16
    config["num_channels"] = 3

    x = torch.randn((8, config["num_channels"], config["image_size"], config["image_size"]))  # Change to (8, 3, 224, 224)
    model = UNetR2D_AF(config)
    total_params = sum(p.numel() for p in model.parameters())
    flops = FlopCountAnalysis(model, x)
    y = model(x)
    # print("Output shape:", y.shape)  # Should print (8, 1, 224, 224) if num_classes = 1
    print(f"Total Parameters:{total_params}" )
    print(f"Total FLOPs:{flops.total() / 10e9} GFLOPs")
