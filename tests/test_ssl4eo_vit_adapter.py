import torch

from adapters.ssl4eo_vit import hls_to_sentinel2, resize_images_and_masks

HLS_TO_S2_BAND_INDEX = {0: 1, 1: 2, 2: 3, 3: 8, 4: 11, 5: 12}


def test_hls_to_sentinel2_places_bands_and_zero_fills_the_rest():
    images = torch.zeros(2, 6, 4, 4)
    for hls_idx in range(6):
        images[:, hls_idx, :, :] = hls_idx + 1.0  # band i filled with value i+1

    s2 = hls_to_sentinel2(images)
    assert s2.shape == (2, 13, 4, 4)

    filled_s2_indices = set(HLS_TO_S2_BAND_INDEX.values())
    for hls_idx, s2_idx in HLS_TO_S2_BAND_INDEX.items():
        assert torch.all(s2[:, s2_idx, :, :] == hls_idx + 1.0)

    for s2_idx in range(13):
        if s2_idx not in filled_s2_indices:
            assert torch.all(s2[:, s2_idx, :, :] == 0.0)


def test_resize_images_and_masks_shapes_and_mask_values_preserved():
    images = torch.rand(2, 6, 512, 512)
    masks = torch.randint(-1, 2, (2, 512, 512))  # values in {-1, 0, 1}

    resized_images, resized_masks = resize_images_and_masks(images, masks, size=224)

    assert resized_images.shape == (2, 6, 224, 224)
    assert resized_masks.shape == (2, 224, 224)
    assert set(resized_masks.unique().tolist()).issubset({-1, 0, 1})
