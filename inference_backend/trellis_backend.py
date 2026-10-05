from .base import (
    BackendCapabilities,
    BaseReconstructor,
    ReconstructionRequest,
    ReconstructionResult,
)

class TrellisReconstructor(BaseReconstructor):
    name = "trellis"
    display_name = "TRELLIS"
    capabilities = BackendCapabilities(
        supports_single_image=True,
        supports_multi_image=True,
        multi_image_modes=("stochastic", "multidiffusion"),
    )
    default_conf = {
        "model_id": "microsoft/TRELLIS-image-large",
        "multiimage_algo": "stochastic",
        "simplify": 0.85,
        "texture_size": 2048,
    }

    def load(self):
        if self._model is not None:
            return

        import sys, os
        from pathlib import Path
        trellis_root = Path(__file__).resolve().parent.parent / "submodules" / "TRELLIS"
        if str(trellis_root) not in sys.path:
            sys.path.insert(0, str(trellis_root))
        os.environ.setdefault("SPCONV_ALGO", "native")

        from trellis.pipelines import TrellisImageTo3DPipeline
        self._model = TrellisImageTo3DPipeline.from_pretrained(self.conf["model_id"])
        self._model.cuda()

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        if self._model is None:
            self.load()

        from trellis.utils import postprocessing_utils

        images = request.images
        algo = request.options.get("multiimage_algo", self.conf["multiimage_algo"])
        if len(images) == 1:
            outputs = self._model.run(images[0], seed=1)
        else:
            outputs = self._model.run_multi_image(images, seed=1, mode=algo)

        glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0],
                                           simplify=self.conf["simplify"],
                                           texture_size=self.conf["texture_size"])
        glb.export(request.output_path)
        return ReconstructionResult(
            glb_path=request.output_path,
            backend_name=self.name,
            num_images_used=len(images),
            raw_meta={"multiimage_algo": algo},
        )