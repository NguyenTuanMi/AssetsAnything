from .base import BaseReconstructor, ReconstructionRequest, ReconstructionResult
from .pixal3d_backend import Pixal3DReconstructor
from .trellis_backend import TrellisReconstructor

REGISTRY: dict[str, type] = {
    "trellis": TrellisReconstructor,
    "pixal3d": Pixal3DReconstructor,
}

_LOADED: dict[str, BaseReconstructor] = {}

def list_backends() -> list[dict]:
    return [
        {
            "name": name,
            "display_name": backend_type.display_name,
            "capabilities": {
                "supports_single_image": backend_type.capabilities.supports_single_image,
                "supports_multi_image": backend_type.capabilities.supports_multi_image,
                "multi_image_modes": list(backend_type.capabilities.multi_image_modes),
            },
            "loaded": name in _LOADED,
        }
        for name, backend_type in REGISTRY.items()
    ]


def get_reconstructor(name: str, conf: dict | None = None) -> BaseReconstructor:
    if name not in REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available: {sorted(REGISTRY)}")

    # Large model backends share one GPU in the current deployment. Keep only
    # one resident so selecting a different backend can release the old model.
    for loaded_name in list(_LOADED):
        if loaded_name != name:
            _LOADED[loaded_name].unload()
            del _LOADED[loaded_name]

    backend = _LOADED.setdefault(name, REGISTRY[name](conf))
    backend.load()
    return backend


__all__ = [
    "BaseReconstructor",
    "ReconstructionRequest",
    "ReconstructionResult",
    "get_reconstructor",
    "list_backends",
]