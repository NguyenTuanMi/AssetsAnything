from .trellis_backend import TrellisReconstructor
from .pixel3d_backend import Pixel3DReconstructor

REGISTRY: dict[str, type] = {
    "trellis": TrellisReconstructor,
    "pixel3d": Pixel3DReconstructor,
}

_LOADED: dict[str, "BaseReconstructor"] = {}   # simple one-slot-at-a-time cache, see §5

def get_reconstructor(name: str, conf: dict | None = None):
    if name not in REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available: {list(REGISTRY)}")
    ...