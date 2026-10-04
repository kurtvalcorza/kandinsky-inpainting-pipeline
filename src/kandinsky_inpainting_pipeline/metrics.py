"""Metrics and evaluation helpers for the Kandinsky 2.2 inpainting pipeline.

Provides:
- Unmasked region preservation metrics (PSNR and SSIM on preserved non-inpaint pixels)
- CLIP scorer for prompt-image alignment
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

SCORER_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SCORER_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"


def _projected(features: Any) -> Any:
    """Return the projected embedding across supported Transformers return types."""
    return features if hasattr(features, "shape") else features.pooler_output


def compute_psnr(
    im1: np.ndarray | Image.Image,
    im2: np.ndarray | Image.Image,
    mask: np.ndarray | Image.Image | None = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    If `mask` is provided, computation is restricted to pixels where mask is True (preserved region).
    """
    a = np.asarray(im1).astype(np.float32)
    b = np.asarray(im2).astype(np.float32)
    if mask is not None:
        m = np.asarray(mask) > 0
        if m.ndim == 2:
            m = m[:, :, None]
        v1 = a[np.broadcast_to(m, a.shape)]
        v2 = b[np.broadcast_to(m, b.shape)]
        if len(v1) == 0:
            return 100.0
        mse = float(np.mean((v1 - v2) ** 2))
    else:
        mse = float(np.mean((a - b) ** 2))

    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def compute_ssim(
    im1: np.ndarray | Image.Image,
    im2: np.ndarray | Image.Image,
    mask: np.ndarray | Image.Image | None = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) in range [-1, 1].

    If `mask` is provided, computes SSIM over the masked region where mask is True (preserved region).
    """
    a = np.asarray(im1).astype(np.float32)
    b = np.asarray(im2).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    if mask is not None:
        m = np.asarray(mask) > 0
        if m.ndim == 2:
            m = m[:, :, None]
        v1 = a[np.broadcast_to(m, a.shape)]
        v2 = b[np.broadcast_to(m, b.shape)]
        if len(v1) == 0:
            return 1.0
        mu1, mu2 = float(np.mean(v1)), float(np.mean(v2))
        s1, s2 = float(np.var(v1)), float(np.var(v2))
        s12 = float(np.mean((v1 - mu1) * (v2 - mu2)))
    else:
        mu1, mu2 = float(np.mean(a)), float(np.mean(b))
        s1, s2 = float(np.var(a)), float(np.var(b))
        s12 = float(np.mean((a - mu1) * (b - mu2)))

    numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)
    return float(numerator / (denominator + 1e-10))


def score_inpainting_preservation(
    original_images: list[Image.Image],
    inpainted_images: list[Image.Image],
    mask_images: list[Image.Image],
) -> dict[str, Any]:
    """Score preservation of unmasked regions.

    mask convention: 1 (or 255) indicates region to inpaint; 0 indicates preserved region.
    Preservation metrics evaluate on the region where mask == 0.
    """
    psnr_scores = []
    ssim_scores = []
    for orig, inpaint, mask in zip(original_images, inpainted_images, mask_images, strict=True):
        m_arr = np.asarray(mask)
        # unmasked region is where mask == 0
        unmasked = (m_arr == 0)
        psnr_scores.append(compute_psnr(orig, inpaint, mask=unmasked))
        ssim_scores.append(compute_ssim(orig, inpaint, mask=unmasked))

    return {
        "n_samples": len(original_images),
        "mean_unmasked_psnr_db": float(np.mean(psnr_scores)),
        "mean_unmasked_ssim": float(np.mean(ssim_scores)),
        "psnr_per_sample": psnr_scores,
        "ssim_per_sample": ssim_scores,
    }


class ClipScorer:
    """Frozen CLIP ViT-B/32 scorer for prompt alignment."""

    def __init__(self, device: str = "cpu", weights_dir: str | None = None):
        self.device = device
        self.weights_dir = weights_dir
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        path = self.weights_dir or SCORER_ID
        self._processor = CLIPProcessor.from_pretrained(path)
        self._model = CLIPModel.from_pretrained(path, torch_dtype=torch.float32).to(self.device).eval()

    def score(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image  # [N, N]
            similarities = torch.diagonal(logits_per_image).cpu().tolist()
        return similarities


def score_generations(scorer: ClipScorer, images: list[Image.Image], prompts: list[str]) -> dict[str, Any]:
    sims = scorer.score(images, prompts)
    return {
        "mean_prompt_similarity": float(np.mean(sims)),
        "prompt_similarities": sims,
    }


REAL_PHOTO_REFERENCE_KIND = "original-photograph reference (per image, own caption only; not a ceiling)"
REAL_PHOTO_REFERENCE_READING = {
    "mean_prompt_similarity": (
        "not an upper bound: a model that repaints the masked region to match the caption can match the caption more "
        "closely than the original photograph, which only has to contain the subject, so outputs can score above it"
    ),
}


def real_photo_reference(scorer: ClipScorer, images: list[Image.Image], prompts: list[str]) -> dict[str, Any]:
    """CLIP prompt similarity of the original, unmasked photographs against their own captions: a reference line, NOT a
    ceiling.

    Each photograph is scored only against its own caption, exactly as an output is, so there is no reference set and no
    photograph is ever compared with itself (the leave-one-out condition holds trivially; contrast the generation
    siblings, whose reference similarity needs the photo excluded from its own reference mean). Outputs can score above
    this line (see `REAL_PHOTO_REFERENCE_READING`)."""
    report = score_generations(scorer, images, prompts)
    report["n_images"] = len(images)
    report["reference_kind"] = REAL_PHOTO_REFERENCE_KIND
    report["reading"] = dict(REAL_PHOTO_REFERENCE_READING)
    report["note"] = (
        "original photographs scored against their own captions only (no reference set, no self-comparison). A reference "
        "line, not a ceiling: inpainted outputs can score above it"
    )
    return report


# The name the generation siblings used; kept so callers written against them get this measure.
real_photo_baseline = real_photo_reference


def count_above_reference(scores: list[float], reference: list[float]) -> dict[str, Any]:
    """How many outputs score above their own original photograph (paired per image)."""
    above = sum(float(s) > float(r) for s, r in zip(scores, reference, strict=True))
    return {"n_above": int(above), "n": len(reference), "text": f"{above} of {len(reference)}"}
