"""Normal-only PatchCore with random or coreset patch selection."""

from math import sqrt
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


class PatchCore:
    """Score patches against normal features from an ImageNet-pretrained backbone.

    Random priority sampling bounds the memory bank; optional farthest-first
    selection picks a coreset from a larger random candidate pool.
    Use a separate saved bank for each MVTec category; no loss or optimizer is used.
    """

    def __init__(self, device="cpu", max_patches=2048, projection_dim=256, seed=42,
                 selection="random"):
        if max_patches < 1 or projection_dim < 1:
            raise ValueError("max_patches and projection_dim must be positive")
        if selection not in ("random", "coreset"):
            raise ValueError("selection must be 'random' or 'coreset'")
        self.device = torch.device(device)
        self.max_patches = max_patches
        self.projection_dim = projection_dim
        self.seed = seed
        self.selection = selection
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.backbone = create_feature_extractor(
            model, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).to(self.device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        generator = torch.Generator().manual_seed(seed)
        self.projection = (
            torch.randn(1536, projection_dim, generator=generator) / sqrt(projection_dim)
        ).to(self.device)
        self.memory_bank = None

    @torch.inference_mode()
    def _embed(self, images):
        features = self.backbone(images.to(self.device))
        shallow = features["layer2"]
        deep = F.interpolate(features["layer3"], size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        patches = torch.cat((shallow, deep), dim=1).permute(0, 2, 3, 1)
        return patches @ self.projection

    @torch.inference_mode()
    def fit(self, loader):
        """Build a bounded bank from normal patches."""
        generator = torch.Generator().manual_seed(self.seed)
        bank = torch.empty((0, self.projection_dim))
        priorities = torch.empty(0)
        pool_size = self.max_patches * (4 if self.selection == "coreset" else 1)
        for batch in loader:
            if "label" in batch and torch.as_tensor(batch["label"]).any():
                raise ValueError("fit accepts normal training images only")
            patches = self._embed(batch["image"]).reshape(-1, self.projection_dim).cpu()
            keys = torch.rand(len(patches), generator=generator)
            keep = torch.topk(keys, min(pool_size, len(keys))).indices
            candidates = torch.cat((bank, patches[keep]))
            candidate_keys = torch.cat((priorities, keys[keep]))
            selected = torch.topk(candidate_keys, min(pool_size, len(candidate_keys))).indices
            bank, priorities = candidates[selected], candidate_keys[selected]
        if not len(bank):
            raise ValueError("training loader is empty")
        if self.selection == "coreset" and len(bank) > self.max_patches:
            vectors = bank.to(self.device)
            nearest = torch.full((len(vectors),), torch.inf, device=self.device)
            indices = torch.empty(self.max_patches, dtype=torch.long, device=self.device)
            choice = torch.zeros((), dtype=torch.long, device=self.device)
            for index in range(self.max_patches):
                indices[index] = choice
                distances = ((vectors - vectors[choice]) ** 2).sum(dim=1)
                nearest = torch.minimum(nearest, distances)
                nearest[choice] = -1
                choice = nearest.argmax()
            bank = bank[indices.cpu()]
        self.memory_bank = bank
        return self

    @torch.inference_mode()
    def predict(self, images):
        """Return CPU image scores [N] and pixel anomaly maps [N,H,W]."""
        if self.memory_bank is None:
            raise RuntimeError("fit or load a memory bank before prediction")
        embeddings = self._embed(images)
        batch_size, patch_height, patch_width, dimension = embeddings.shape
        patches = embeddings.reshape(-1, dimension)
        bank = self.memory_bank.to(self.device)
        distances = torch.cat([
            torch.cdist(chunk, bank).min(dim=1).values
            for chunk in patches.split(1024)
        ]).reshape(batch_size, 1, patch_height, patch_width)
        scores = distances.flatten(1).amax(dim=1)
        maps = F.interpolate(distances, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return scores.cpu(), maps[:, 0].cpu()

    def save(self, path):
        if self.memory_bank is None:
            raise RuntimeError("fit a memory bank before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "max_patches": self.max_patches,
            "projection_dim": self.projection_dim,
            "seed": self.seed,
            "selection": self.selection,
            "projection": self.projection.cpu(),
            "memory_bank": self.memory_bank,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(device, state["max_patches"], state["projection_dim"],
                    state["seed"], state.get("selection", "random"))
        model.projection = state["projection"].to(model.device)
        model.memory_bank = state["memory_bank"]
        return model
