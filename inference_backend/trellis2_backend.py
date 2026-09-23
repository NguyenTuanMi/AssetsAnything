from .base import (
    BackendCapabilities,
    BaseReconstructor,
    ReconstructionRequest,
    ReconstructionResult,
)
import o_voxel

class TrellisReconstructor(BaseReconstructor):
    name = "trellis2"
    display_name = "TRELLIS2"
    capabilities = BackendCapabilities(
        supports_single_image=True,
        supports_multi_image=True,
        multi_image_modes=("stochastic", "multidiffusion"),
    )
    default_conf = {
        "model_id": "microsoft/TRELLIS.2-4B",
        "multiimage_algo": "stochastic",
        "simplify": 0.85,
        "texture_size": 2048,
    }

    def load(self):
        if self._model is not None:
            return

        import sys, os
        from pathlib import Path
        trellis_root = Path(__file__).resolve().parent.parent / "submodules" / "TRELLIS.2"
        if str(trellis_root) not in sys.path:
            sys.path.insert(0, str(trellis_root))

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        self._model = Trellis2ImageTo3DPipeline.from_pretrained(self.conf["model_id"])
        self._model.cuda()

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        if self._model is None:
            self.load()

        from trellis2.utils import render_utils

        images = request.images
        algo = request.options.get("multiimage_algo", self.conf["multiimage_algo"])
        if len(images) == 1:
            mesh = self._model.run(images[0])[0]
        else:
            mesh = self._model.run_multi_image(images, seed=1, mode=algo)

        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   1000000,
            texture_size        =   4096,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        glb.export(request.output_path)
        return ReconstructionResult(
            glb_path=request.output_path,
            backend_name=self.name,
            num_images_used=len(images),
            raw_meta={"multiimage_algo": algo},
        )