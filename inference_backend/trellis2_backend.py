from .base import (
    BackendCapabilities,
    BaseReconstructor,
    ReconstructionRequest,
    ReconstructionResult,
)


class Trellis2Reconstructor(BaseReconstructor):
    name = "trellis2"
    display_name = "TRELLIS2"
    capabilities = BackendCapabilities(
        supports_single_image=True,
        supports_multi_image=False,
    )
    default_conf = {
        "model_id": "microsoft/TRELLIS.2-4B",
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

        import o_voxel

        images = request.images
        if len(images) != 1:
            raise ValueError("TRELLIS.2 currently supports exactly one image.")

        mesh = self._model.run(images[0], seed=1)[0]

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
        glb.export(request.output_path, extension_webp=True)
        return ReconstructionResult(
            glb_path=request.output_path,
            backend_name=self.name,
            num_images_used=len(images),
            raw_meta={"pipeline": "Trellis2ImageTo3DPipeline"},
        )