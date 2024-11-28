import torch
import torch.nn as nn
import timm
from model import attention_gate
from fvcore.nn import FlopCountAnalysis
from transformers import ViTModel, EncoderDecoderModel
from torchvision import models

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



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze step: Global average pooling
        y = self.global_avg_pool(x)
        
        # Excitation step: Fully connected layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Scale the input features
        return x * y


class UNetR2D(nn.Module):
    def __init__(self, cg, vit_model="google/vit-base-patch16-224-in21k", pretrained=True, num_classes=1):
        super().__init__()
        self.cg = cg
        
        # Load pre-trained Vision Transformer (ViT) model
        self.vit = ViTModel.from_pretrained(vit_model, output_hidden_states=True)
        
        # CNN Decoder
        self.d1 = DeconvBlock(cg["hidden_dim"], 512)
        self.s1 = nn.Sequential(
            DeconvBlock(cg["hidden_dim"], 512),
            ConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512 + 512, 512),
            ConvBlock(512, 512)
        )

        # Decoder 2
        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(cg["hidden_dim"], 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256 + 256, 256),
            ConvBlock(256, 256)
        )
    
        # Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(cg["hidden_dim"], 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128 + 128, 128),
            ConvBlock(128, 128)
        )

        # Decoder 4
        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ConvBlock(64, 64)
        )

        # Final Layer
        self.output = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Get hidden states from ViT model
        vit_outputs = self.vit(pixel_values=x)
        hidden_states = vit_outputs.hidden_states  # Tuple of hidden states at each layer

        # Reshape the hidden states from (batch_size, sequence_length, hidden_dim) to (batch_size, hidden_dim, H, W)
        # The sequence_length should be num_patches + 1 (for the class token), so we exclude the class token and reshape the patches
        batch_size, seq_length, hidden_dim = hidden_states[-1].size()
        
        patch_size = self.cg["patch_size"]
        grid_size = int((seq_length - 1) ** 0.5)  # For 197 tokens, grid_size would be 14 for 14x14 patches

        # Remove the class token and reshape
        z12 = hidden_states[-1][:, 1:, :].reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)

        # Decoder 1
        x = self.d1(z12)
        z9 = hidden_states[9][:, 1:, :].reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # Decoder 2
        x = self.d2(x)
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


# class UNetR2D(nn.Module):
#     def __init__(self, cg, vit_model="google/vit-base-patch16-224-in21k", pretrained=True, num_classes=1):
#         super().__init__()
#         self.cg = cg
#         #self.vit = timm.create_model(vit_model, pretrained=pretrained)
#         #self.vit = ViTFeatureExtractor.from_pretrained(vit_model)
#         #self.vit =  models.vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1")
#         #print(self.vit.encoder)

#         self.patch_size = cg["patch_size"] 
#         self.hidden_dim = cg["hidden_dim"]
#         self.num_patches = cg["num_patches"]
#         self.patch_embed = nn.Linear(
#             self.patch_size * self.patch_size * cg["num_channels"], self.hidden_dim 
#         )  

#         self.positions = torch.arange(start=0, end=cg["num_patches"], step=1, dtype=torch.int32)
#         self.pos_embed = nn.Embedding(self.num_patches, self.hidden_dim)

#         # self.vit.encoder_layers = nn.ModuleList([
#         #     nn.TransformerEncoderLayer(
#         #         d_model = cg['hidden_dim'] ,
#         #         nhead = cg["num_heads"],
#         #         dim_feedforward= cg["mlp_dim"],
#         #         dropout=cg["dropout_rate"],
#         #         activation="gelu",
#         #         batch_first=True  
#         #     ) for _ in range(cg["num_layers"])
#         # ] )

#         # CNN Decoder
#         # Decoder 1
#         self.d1 = DeconvBlock(cg["hidden_dim"], 512)
#         self.s1 = nn.Sequential(
#             DeconvBlock(cg["hidden_dim"], 512),
#             ConvBlock(512, 512)
#         )
#         self.c1 = nn.Sequential(
#             ConvBlock(512+512, 512),
#             ConvBlock(512, 512)
#         )

#         # Decoder 2
#         self.d2 = DeconvBlock(512, 256)
#         self.s2 = nn.Sequential(
#             DeconvBlock(cg["hidden_dim"], 256),
#             ConvBlock(256, 256),
#             DeconvBlock(256, 256),
#             ConvBlock(256, 256)
#         )
#         self.c2 = nn.Sequential(
#             ConvBlock(256+256, 256),
#             ConvBlock(256, 256)
#         )
    
#         # Decoder 3
#         self.d3 = DeconvBlock(256, 128)
#         self.s3 = nn.Sequential(
#             DeconvBlock(cg["hidden_dim"], 128),
#             ConvBlock(128, 128),
#             DeconvBlock(128, 128),
#             ConvBlock(128, 128),
#             DeconvBlock(128, 128),
#             ConvBlock(128, 128)
#         )
#         self.c3 = nn.Sequential(
#             ConvBlock(128+128, 128),
#             ConvBlock(128, 128)
#         )

#         # Decoder 4
#         self.d4 = DeconvBlock(128, 64)
#         self.s4 = nn.Sequential(
#             ConvBlock(3, 64),
#             ConvBlock(64, 64)
#         )
#         self.c4 = nn.Sequential(
#             ConvBlock(64+64, 64),
#             ConvBlock(64, 64)
#         )

#         # Final Layer
#         self.output = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)


#     def forward(self, x):
#         # Debugging: check the shape of the input tensor
#         #print("Input shape before patching:", x.shape)  # Expected shape: (B, C, H, W)

#         # # Ensure x is of shape (B, C, H, W)
#         # if x.dim() != 4:
#         #     raise ValueError(f"Expected input to have 4 dimensions (batch, channels, height, width). Got: {x.dim()}")

#         # Unpack the batch size, channels, height, and width

#         batch_size, _, H, W = x.size()
#         patch_size = self.cg["patch_size"] 
#         num_patches = self.cg["num_patches"] 
#         x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         x = x.contiguous().view(batch_size, -1, patch_size * patch_size * self.cg["num_channels"] )

#         patch_embed = self.patch_embed(x)
#         positions = torch.arange(num_patches, dtype=torch.long, device=x.device)
#         pos_embed = self.pos_embed(positions)
#         pos_embed = self.pos_embed(positions).unsqueeze(0).expand(x.size(0), -1, -1)
#         x = patch_embed + pos_embed

#        # Transformer Encoder
#         skip_connections_ind = [3, 6, 9, 12]  # indices of layers where skip connections are added
#         skip_connections = []
#         for i in range(self.cg["num_layers"]):
#             layer = self.vit.encoder_layers[i]
#             x = layer(x)
#             if (i+1) in skip_connections_ind:
#                 skip_connections.append(x)
#             #print(f"Skip Connections:{len(skip_connections)} ")
#         # CNN Decoders
#         z3, z6, z9, z12 = skip_connections
#         batch = x.shape[0]
#         #z0 = x.view((batch_size, self.cg["num_channels"], self.cg["image_size"], self.cg["image_size"]))
#         z0 = x.view((batch, self.cg["num_channels"], self.cg["image_size"], self.cg["image_size"]))

#         shape = (batch_size, self.cg["hidden_dim"], self.cg["patch_size"], self.cg["patch_size"])
#         z3 = z3.view(shape)
#         z6 = z6.view(shape)
#         z9 = z9.view(shape)
#         z12 = z12.view(shape)

#         # Decoder 1
#         x = self.d1(z12)
#         s = self.s1(z9)
#         x = torch.cat([x, s], dim=1)
#         x = self.c1(x)
    
#         # Decoder 2
#         x = self.d2(x)
#         s = self.s2(z6)
#         x = torch.cat([x, s], dim=1)
#         x = self.c2(x)

#         # Decoder 3
#         x = self.d3(x)
#         s = self.s3(z3)
#         x = torch.cat([x, s], dim=1)
#         x = self.c3(x)

#         # Decoder 4
#         x = self.d4(x)
#         s = self.s4(z0)
#         x = torch.cat([x, s], dim=1)
#         x = self.c4(x)

#         # Final Layer
#         output = self.output(x)

#         return output


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
    model = UNetR2D(config)
    total_params = sum(p.numel() for p in model.parameters())
    flops = FlopCountAnalysis(model, x)
    y = model(x)
    # print("Output shape:", y.shape)  # Should print (8, 1, 224, 224) if num_classes = 1
    print(f"Total Parameters:{total_params}" )
    #print(f"Total FLOPs:{flops.total()} GFLOPs")
