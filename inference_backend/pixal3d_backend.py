from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .base import (
	BackendCapabilities,
	BaseReconstructor,
	ReconstructionRequest,
	ReconstructionResult,
)


class Pixal3DReconstructor(BaseReconstructor):
	"""Run Pixal3D in its own conda environment.

	Pixal3D currently has binary dependencies that are not compatible with the
	TRELLIS environment. The subprocess boundary keeps those imports and CUDA
	extensions out of this FastAPI process. A persistent Pixal3D worker can
	replace this adapter later without changing the server contract.
	"""

	name = "pixal3d"
	display_name = "Pixal3D"
	capabilities = BackendCapabilities(
		supports_single_image=True,
		supports_multi_image=False,
	)
	default_conf = {
		"conda_env": os.environ.get("PIXAL3D_CONDA_ENV", "pixal3d"),
		"repo_dir": str(Path(__file__).resolve().parent.parent / "submodules" / "Pixal3D"),
		"script": "inference.py",
		"low_vram": True,
		"resolution": 1024,
		"timeout_seconds": 1800,
	}

	def load(self) -> None:
		repo_dir = Path(self.conf["repo_dir"])
		script_path = repo_dir / self.conf["script"]
		if not script_path.is_file():
			raise RuntimeError(
				f"Pixal3D submodule is not initialized or is missing "
				f"{script_path}. Run: git submodule update --init --recursive"
			)

		# No model is imported here. The model is loaded by the isolated
		# Pixal3D process for each invocation until a persistent worker exists.
		self._model = script_path

	def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
		if self._model is None:
			self.load()
		if len(request.images) != 1:
			raise ValueError("Pixal3D currently supports exactly one image.")

		repo_dir = Path(self.conf["repo_dir"])
		output_path = Path(request.output_path).resolve()
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with tempfile.TemporaryDirectory(prefix="pixal3d_input_") as temp_dir:
			input_path = Path(temp_dir) / "input.png"
			request.images[0].convert("RGBA").save(input_path, format="PNG")

			command = [
				"conda",
				"run",
				"--no-capture-output",
				"-n",
				self.conf["conda_env"],
				"python",
				str(self._model),
				"--image",
				str(input_path),
				"--output",
				str(output_path),
			]
			if self.conf["low_vram"]:
				command.append("--low_vram")
			if self.conf.get("resolution"):
				command.extend(["--resolution", str(self.conf["resolution"])])

			print(f"[pixal3d] running isolated environment {self.conf['conda_env']}")
			try:
				subprocess.run(
					command,
					cwd=repo_dir,
					check=True,
					timeout=self.conf["timeout_seconds"],
				)
			except subprocess.TimeoutExpired as error:
				raise RuntimeError("Pixal3D inference timed out.") from error
			except subprocess.CalledProcessError as error:
				raise RuntimeError(
					f"Pixal3D inference failed with exit code {error.returncode}."
				) from error

		if not output_path.is_file():
			raise RuntimeError(f"Pixal3D did not produce {output_path}.")

		return ReconstructionResult(
			glb_path=str(output_path),
			backend_name=self.name,
			num_images_used=1,
			raw_meta={
				"execution": "subprocess",
				"conda_env": self.conf["conda_env"],
				"resolution": self.conf.get("resolution"),
			},
		)
