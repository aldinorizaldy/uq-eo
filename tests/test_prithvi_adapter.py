import torch

from adapters.prithvi import TTA_TRANSFORMS


def test_tta_transforms_are_true_inverses():
    x = torch.randn(2, 3, 8, 8)
    for forward_tf, inverse_tf in TTA_TRANSFORMS:
        reconstructed = inverse_tf(forward_tf(x))
        torch.testing.assert_close(reconstructed, x)


def test_tta_transforms_actually_change_the_tensor_except_identity():
    # sanity check that the non-identity transforms aren't accidentally no-ops
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    identity_pair, *rest = TTA_TRANSFORMS
    assert torch.equal(identity_pair[0](x), x)
    for forward_tf, _ in rest:
        assert not torch.equal(forward_tf(x), x)
