#!/bin/bash
set -e
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "trellis" ]]; then
    echo "No conda environment is activated or conda environment is not corrected."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate trellis
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
fi  

CUDA_VISIBLE_DEVICES=1 python inference_server/inference_server.py --host 0.0.0.0 --port 8000