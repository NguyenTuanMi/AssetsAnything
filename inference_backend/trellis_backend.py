from .base import BaseReconstructor, ReconstructionResult

class TrellisReconstructor(BaseReconstructor):
    name = "trellis"
    default_conf = {"multiimage_algo": "stochastic", "sparse_steps": 12, "slat_steps": 12}
    supports_multi_image = True

    def load(self):
        import sys, os
        from pathlib import Path
        trellis_root = Path(__file__).resolve().parent.parent / "submodules" / "TRELLIS"
        if str(trellis_root) not in sys.path:
            sys.path.insert(0, str(trellis_root))
        os.environ.setdefault("SPCONV_ALGO", "native")

        from trellis.pipelines import TrellisImageTo3DPipeline
        self._model = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self._model.cuda()

    def run(self, images, **kwargs) -> ReconstructionResult:
        from trellis.utils import postprocessing_utils
        import uuid, os

        algo = kwargs.get("multiimage_algo", self.conf["multiimage_algo"])
        if len(images) == 1:
            outputs = self._model.run(images[0], seed=1)
        else:
            outputs = self._model.run_multi_image(images, seed=1, mode=algo)

        glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0],
                                           simplify=0.95, texture_size=1024)
        out_path = f"/tmp/{uuid.uuid4()}.glb"   # OUTPUT_DIR wiring stays in inference_server.py
        glb.export(out_path)
        return ReconstructionResult(
            glb_path=out_path, supports_multi_image=(len(images) > 1),
            backend_name=self.name, raw_meta={"algo": algo},
        )