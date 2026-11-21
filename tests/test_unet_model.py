import torch

from adapters.unet_model import UNet


def test_unet_output_shape_matches_input_spatial_size():
    model = UNet(in_channels=6, n_classes=2, base_channels=8)
    x = torch.randn(2, 6, 64, 64)  # 64 divisible by 8 (3 pooling stages)
    out = model(x)
    assert out.shape == (2, 2, 64, 64)
