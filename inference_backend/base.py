from dataclasses import dataclass
from PIL import Image

@dataclass
class ReconstructionResult:
    glb_path: str
    supports_multi_image: bool   # did this run actually fuse multiple views, or just use one?
    backend_name: str
    raw_meta: dict               # backend-specific extras (sampler steps used, etc.) — never load-bearing outside the backend

class BaseReconstructor(ABC):
    """One instance = one loaded model, resident on the GPU until unload()."""

    name: str                # unique registry key, e.g. "trellis", "pixel3d"
    default_conf: dict = {}
    supports_single_image: bool = True
    supports_multi_image: bool = False

    def __init__(self, conf: dict | None = None):
        self.conf = {**self.default_conf, **(conf or {})}
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """Import + construct the model. MUST be lazy — nothing backend-specific
        gets imported at module top level, so a broken/uninstalled backend's env
        can't take down server startup for every other backend."""

    @abstractmethod
    def run(self, images: list[Image.Image], **kwargs) -> ReconstructionResult:
        ...

    def unload(self) -> None:
        """Default: drop the reference and clear CUDA cache. Override if a
        backend needs extra cleanup."""
        import torch
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()