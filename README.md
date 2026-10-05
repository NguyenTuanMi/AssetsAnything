<h1 align="center">AssetsAnything<br>A general framework for all-purpose 3D Generation</h1>

AssetsAnything is an Image-to-3D framework that allows users to render 3D Assets from images using famous backend models, such as TRELLIS, Pixal3D, etc. Users can decide to use their own images or they can input the prompts describing their wanted object and AssetsAnything will scrape images of that specific object for reconstruction. Currently this framework only supports TRELLIS, but more models are being incorporated! Stay tune!

<!-- Features -->
## 🌟 Features

- **High Quality**: It produces diverse 3D assets at high quality with intricate shape and texture details.
- **Versatility**: It takes text or image prompts and rewrite/preprocess them before generating various final 3D representations.
- **Flexible Editing**: It allows users to select images for reconstruction and edit the dimensions of 3D Asset before exporting.

<!-- Installation -->
## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**. 
- **Hardware**: At lest 2 NVIDIA GPUs with at least 16GB of memory is necessary. The code has been verified on 2 NVIDIA RTX 6000 Ada Generations.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone --recurse-submodules https://github.com/NguyenTuanMi/AssetsAnything.git
    cd AssetsAnything
    ```

2. Install the dependencies:
    
    Create a new conda environment named `trellis` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```

3. For prompt restructure and images processing, Qwen 2.5 VL is launched locally on a second GPU in a different environment. 
    
    Create a new conda environment named `qwen_mass_estimator` and install the dependencies:
    ```sh
    conda create --name qwen_mass_estimator python=3.10
    pip install vllm openai
    ```

An environment.yaml file is included for reference of installed packages in the conda environment, but not recommended to use for installation. 

<!-- Deployment -->
## 🏋️‍♂️ Deployment

1. Change access permission for launch scripts:
    ```sh
    chmod +x scripts/*
    ```

2. Add SERP_API key and source the environment:
    
    Please visit the [SERPAPI](https://serpapi.com/) and create an account to get access to a free tier api key. After that, please paste the key into the .env file, then proceed with steps below: 
    ```sh
    source .env
    ```

3. On one terminal, run this command to launch the VLLM for Qwen: 
    ```sh
    ./scripts/deploy/launch_qwen.sh
    ```

4. On anotherr terminal, run this command to launch the backend:
    ```sh
    ./scripts/deploy/launch_backend.sh
    ```

    The management API now starts model workers on demand and keeps only one
    worker process active on the configured GPU. Set worker interpreters when
    models use separate Pixi or conda environments; otherwise the current
    Python interpreter is used:
    ```sh
    export TRELLIS_WORKER_PYTHON=/path/to/trellis/.pixi/envs/default/bin/python
    export TRELLIS2_WORKER_PYTHON=/path/to/trellis2/.pixi/envs/default/bin/python
    export PIXAL3D_WORKER_PYTHON=/path/to/pixal3d/.pixi/envs/default/bin/python
    ```

    `TRELLIS_WORKER_PORT`, `TRELLIS2_WORKER_PORT`, and `PIXAL3D_WORKER_PORT`
    optionally override the local worker ports. The supervisor terminates the
    active worker's process group and waits for it to exit before starting a
    different backend.

5. On another terminal, launch the UI: 
    ```sh
    cd client && python3 -m http.server 5500
    ```

6. After that, please navigate to [this address](http://localhost:5500/) on any web browser, such as Firefox, Chrome, Bing, etc and start playing around with it!
