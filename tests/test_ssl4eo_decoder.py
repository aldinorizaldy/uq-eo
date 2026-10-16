import torch

from adapters.ssl4eo_decoder import DecoderEnsemble, PatchTokenDecoder


def test_patch_token_decoder_output_shape():
    decoder = PatchTokenDecoder(embed_dim=384, patch_grid_size=14, n_classes=2)
    patch_tokens = torch.randn(2, 196, 384)  # batch=2, 14x14 patches, ViT-S embed dim
    out = decoder(patch_tokens)
    assert out.shape == (2, 2, 224, 224)


def test_decoder_ensemble_stacks_member_outputs():
    decoders = [PatchTokenDecoder(embed_dim=384, patch_grid_size=14, n_classes=2) for _ in range(3)]
    ensemble = DecoderEnsemble(decoders)
    patch_tokens = torch.randn(2, 196, 384)
    out = ensemble(patch_tokens)
    assert out.shape == (3, 2, 2, 224, 224)
