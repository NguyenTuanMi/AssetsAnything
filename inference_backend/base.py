from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class BackendCapabilities:
    supports_single_image: bool = True
    supports_multi_image: bool = False
    multi_image_modes: tuple[str, ...] = ()


@dataclass
class ReconstructionRequest:
    images: list[Image.Image]
    output_path: str
    options: dict[str, Any] = field(default_factory=dict)

@dataclass
class ReconstructionResult:
    glb_path: str
    backend_name: str
    num_images_used: int
    raw_meta: dict[str, Any] = field(default_factory=dict)

class BaseReconstructor(ABC):
    """One instance represents one lazily loaded model."""

    name: str
    display_name: str
    capabilities = BackendCapabilities()
    default_conf: dict = {}

    def __init__(self, conf: dict | None = None):
        self.conf = {**self.default_conf, **(conf or {})}
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """Import + construct the model. MUST be lazy — nothing backend-specific
        gets imported at module top level, so a broken/uninstalled backend's env
        can't take down server startup for every other backend."""

    @abstractmethod
    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        ...

    def unload(self) -> None:
        """Default: drop the reference and clear CUDA cache. Override if a
        backend needs extra cleanup."""
        import torch
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()