"""Adapter for Prithvi-EO-2.0-300M-TL (ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
via TerraTorch, used for the milestone-4 model-agnostic check. Same Adapter contract as
ssl4eo_vit.py so the rest of the eouq pipeline is unchanged when swapping backbones.

Unlike SSL4EO-S12, this checkpoint is already fine-tuned end-to-end for flood
segmentation -- there's no decoder left to train ourselves, so there's no
deep ensemble of independently-trained members either. The predictive-
uncertainty (U_T) signal instead comes from test-time augmentation (TTA):
the same frozen model is run on 5 geometric views of each chip (identity,
horizontal/vertical flip, 90/180-degree rotation), predictions are
inverse-transformed back to the original orientation, and the 5 results are
treated as "ensemble members" by every downstream eouq layer exactly like a
real trained ensemble would be -- the UQ pipeline itself needs no changes
to accept this. See NOTES.md for why TTA was chosen over MC-dropout (this
checkpoint's dropout availability is unconfirmed; TTA works on any frozen
model by construction).

Loading rewritten after two real failed runs: `BACKBONE_REGISTRY.build()`
was the wrong approach entirely -- this checkpoint's config.json uses a
TerraTorch task schema (`model.class_path =
"terratorch.tasks.SemanticSegmentationTask"`, backbone=
"prithvi_eo_v2_300_tl", decoder="UperNetDecoder", num_classes=2), not a
plain timm model. Confirmed the real loading path by reading the HF repo's
own `inference.py` reference script verbatim:
  - Load via `terratorch.cli_tools.LightningInferenceModel.from_config(
    config_path, checkpoint_path)`, where both files are downloaded from
    the HF repo (config.yaml + Prithvi-EO-V2-300M-TL-Sen1Floods11.pt),
    not via BACKBONE_REGISTRY.
  - Forward signature is `model(x, temporal_coords=None,
    location_coords=...)`, returning a wrapper object -- logits are at
    `.output`, not the return value directly.
  - Normalization is handled by `lightning_model.datamodule.test_transform`
    + `.aug`, reused here rather than hand-rolled, since those carry the
    actual training-time per-band statistics.
  - num_classes=2 confirmed directly from config.json -- EVENT_CLASS_INDEX
    below (index 1 of 2, i.e. -1) was already right.
  - Input band indices [1,2,3,8,11,12] confirmed exactly matching what
    `data/sen1floods11.py` already selects.
Still unverified (not yet reached by a real run): whether
`lightning_model.model.encoder(x)` is the right call for the embedding
(U_E) signal -- wrapped in a try/except that falls back to no embeddings
(same graceful degradation already used for the U-Net adapter) rather than
crashing the whole extraction over a secondary signal.
"""
from __future__ import annotations

import numpy as np
import torch

from adapters.base import Adapter, ExtractedOutputs

EVENT_CLASS_INDEX = -1  # index of the "water" class within the model's 2 output channels
CHECKPOINT_FILENAME = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
CONFIG_FILENAME = "config.yaml"


def _flip_h(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _flip_v(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=(-2, -1))


def _rot90_inv(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=-1, dims=(-2, -1))


def _rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=2, dims=(-2, -1))


# (forward transform, inverse transform) pairs -- identity, flips, and 180-deg
# rotation are self-inverse; 90-deg rotation is not.
TTA_TRANSFORMS = [
    (lambda x: x, lambda x: x),
    (_flip_h, _flip_h),
    (_flip_v, _flip_v),
    (_rot90, _rot90_inv),
    (_rot180, _rot180),
]


class PrithviAdapter(Adapter):
    def __init__(
        self,
        checkpoint: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        device: str = "cuda",
        local_dir: str | None = "/data/prithvi_sen1floods11",
        use_location_coords: bool = True,
    ):
        """use_location_coords: the HF reference inference.py passes each chip's
        lon/lat, but the checkpoint's own config.yaml says it was fine-tuned with
        `use_metadata: false` -- i.e. without them. Set False to match training
        conditions instead of the reference script (diagnostic for the low F1)."""
        self.checkpoint = checkpoint
        self.device = device
        self.local_dir = local_dir
        self.use_location_coords = use_location_coords
        self._lightning_model = None

    def _resolve_files(self) -> tuple:
        """Prefer pre-downloaded local copies of config.yaml + the .pt checkpoint.

        Downloading via huggingface_hub inside the container failed on a real run
        (job 867649) in the HTTP layer itself -- httpx2's response decompressor
        called its compression backend with a keyword argument the installed
        backend doesn't accept (`process() takes no keyword arguments`). Rather
        than pin yet another library, the files are fetched once on the login
        node with plain curl (see NOTES.md) and read from `local_dir` here;
        hf_hub_download is only a fallback if they're missing.
        """
        import os

        if self.local_dir:
            config_path = os.path.join(self.local_dir, CONFIG_FILENAME)
            checkpoint_path = os.path.join(self.local_dir, CHECKPOINT_FILENAME)
            if os.path.exists(config_path) and os.path.exists(checkpoint_path):
                return config_path, checkpoint_path

        from huggingface_hub import hf_hub_download

        return (
            hf_hub_download(repo_id=self.checkpoint, filename=CONFIG_FILENAME),
            hf_hub_download(repo_id=self.checkpoint, filename=CHECKPOINT_FILENAME),
        )

    def _load_model(self):
        from terratorch.cli_tools import LightningInferenceModel

        config_path, checkpoint_path = self._resolve_files()

        lightning_model = LightningInferenceModel.from_config(config_path, checkpoint_path)
        lightning_model.model.eval()
        lightning_model.model.to(self.device)
        return lightning_model

    def _preprocess(self, lightning_model, image: torch.Tensor) -> torch.Tensor:
        """image: (C, H, W) float tensor already in [0, 1] reflectance.
        Mirrors the HF repo's inference.py transform sequence exactly:
        channels-last numpy -> datamodule's test_transform -> aug -> tensor."""
        # .contiguous(): TTA's rot90 views come back as non-contiguous
        # transposed views, which survive the numpy round-trip and crash
        # kornia's normalize (it uses .view(); job 884086). The identity/flip
        # views were already contiguous, which is why only rotation broke.
        array = image.contiguous().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        transformed = lightning_model.datamodule.test_transform(image=array)
        if isinstance(transformed.get("image"), torch.Tensor):
            transformed["image"] = transformed["image"].contiguous()
        tensor = lightning_model.datamodule.aug(transformed)["image"]
        return tensor.to(self.device)

    @staticmethod
    def _ensure_batched(x: torch.Tensor) -> torch.Tensor:
        """The reference inference.py passes `datamodule.aug(...)["image"]` to the
        model as-is -- aug already returns a batched (1, C, H, W) tensor and the
        Prithvi backbone inserts the time dimension itself. An earlier version
        of this adapter added batch+time dims on top, giving a 6-D tensor that
        crashed patch_embed (job 884038, "too many values to unpack (expected
        5)"). Only add a batch dim if aug happens to return an unbatched one."""
        return x.unsqueeze(0) if x.dim() == 3 else x

    def _forward_one(self, lightning_model, image: torch.Tensor, location_coords: torch.Tensor) -> torch.Tensor:
        """image: (C, H, W). Returns (n_classes, H, W) logits for this one chip."""
        x = self._ensure_batched(self._preprocess(lightning_model, image))
        coords = location_coords.unsqueeze(0) if location_coords is not None else None
        output = lightning_model.model(x, temporal_coords=None, location_coords=coords)
        return output.output.squeeze(0)  # (n_classes, H, W)

    def _try_embedding(self, lightning_model, image: torch.Tensor) -> np.ndarray | None:
        try:
            x = self._ensure_batched(self._preprocess(lightning_model, image))
            # lightning_model.model is a task wrapper whose forward delegates to
            # .model (a terratorch PixelWiseModel) -- the encoder lives there
            # (per the job 884038 traceback: trainers/base.py -> pixel_wise_model.py).
            pixel_wise_model = lightning_model.model.model
            features = pixel_wise_model.encoder(x)
            if isinstance(features, (list, tuple)):
                features = features[-1]  # deepest encoder stage
            if features.dim() == 3:  # (B, n_tokens, D) -> mean over tokens, keep D
                features = features.mean(dim=1)
            elif features.dim() == 4:  # (B, D, H, W) -> mean over space, keep D
                features = features.mean(dim=(2, 3))
            return features.squeeze(0).detach().cpu().numpy()
        except Exception:
            return None

    def extract(self, dataloader) -> ExtractedOutputs:
        if self._lightning_model is None:
            self._lightning_model = self._load_model()
        lightning_model = self._lightning_model

        all_probs, all_embeddings, all_coords, all_chip_ids, all_masks = [], [], [], [], []
        embeddings_available = True
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)  # (batch, C, H, W), already [0,1]
                coords = batch["coords"]  # (batch, 2) = [lat, lon]
                batch_size = images.shape[0]

                for i in range(batch_size):
                    image = images[i]
                    location_coords = (
                        torch.tensor([coords[i, 1].item(), coords[i, 0].item()], device=self.device)
                        if self.use_location_coords
                        else None
                    )  # [lon, lat], matching the reference script's src.lnglat() order

                    member_logits = []
                    for forward_tf, inverse_tf in TTA_TRANSFORMS:
                        logits = self._forward_one(lightning_model, forward_tf(image), location_coords)
                        member_logits.append(inverse_tf(logits))
                    member_logits = torch.stack(member_logits, dim=0)  # (n_members, n_classes, H, W)
                    member_probs = torch.softmax(member_logits, dim=1).cpu().numpy()
                    all_probs.append(member_probs[:, None, ...])  # add sample axis

                    if embeddings_available:
                        embedding = self._try_embedding(lightning_model, image)
                        if embedding is None:
                            embeddings_available = False
                        else:
                            all_embeddings.append(embedding)

                all_coords.append(coords.cpu().numpy())
                all_chip_ids.extend(batch["chip_id"])
                all_masks.append(batch["mask"].cpu().numpy())

        probs = np.concatenate(all_probs, axis=1)  # (n_members, n_samples, n_classes, H, W)
        coords = np.concatenate(all_coords, axis=0)
        masks = np.concatenate(all_masks, axis=0)
        embeddings = np.stack(all_embeddings, axis=0) if embeddings_available and all_embeddings else None

        return ExtractedOutputs(
            probs=probs,
            embeddings=embeddings,
            coords=coords,
            extra={"chip_id": np.array(all_chip_ids), "mask": masks},
        )
