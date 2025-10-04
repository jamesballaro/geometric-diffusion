# Geometric Diffusion: Probability Density Geodesics in Diffusion Latent Space

A unified implementation combining two cutting-edge research approaches for understanding and manipulating diffusion model latent spaces through geometric analysis.

## Overview

This repository integrates and adapts two seminal research papers to create a comprehensive framework for geometric analysis of diffusion model latent spaces:

1. **[Probability Density Geodesics in Image Diffusion Latent Space](https://arxiv.org/abs/2504.06675)** (CVPR 2025) - Yu et al.
2. **[Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry](https://arxiv.org/pdf/2307.12868)** - Park et al.

The implementation provides a sequential algorithmic process that combines probability density geodesic computation with pullback metric analysis for enhanced image interpolation and semantic editing capabilities.

## Key Features

### **Probability Density Geodesics** (Paper 1)
- **Geodesic Computation**: Computes shortest paths in diffusion latent space where distance is inversely proportional to probability density
- **Boundary Value Problem (BVP) Algorithm**: Solves initial and boundary value problems for geodesic pathfinding
- **Spherical Cubic Splines**: Implements smooth interpolation on hyperspheres with proper geometric constraints
- **Bisection Sampling**: Course-to-fine optimization strategy for control point refinement

### **Pullback Metric Analysis** (Paper 2)
- **Local Latent Basis Discovery**: Derives local basis vectors through SVD of Jacobian matrices
- **Semantic Editing**: Enables meaningful image manipulation by traversing along discovered basis vectors
- **Feature Space Mapping**: Bridges latent space (𝒳) and feature space (ℋ) through pullback metrics
- **Single-Timestep Editing**: Performs edits at specific diffusion timesteps without multi-step optimization

### **Integrated Workflow**
1. **Geodesic Optimization**: Find optimal interpolation paths between two images
2. **Semantic Enhancement**: Apply pullback metric analysis to selected points along the geodesic
3. **Controlled Generation**: Generate semantically meaningful intermediate images

## Example Use Case:
1. **Interpolate**: Find two images which you would like to hybridise
2. **Selection**: Choose any number of intermediate images which you would like to modify semantically
3. **Edit**: Use the meaningful latent directions derived from the pullback metric to modify that selection

## Architecture

```
src/
├── main.py               # Main execution script
├── pipeline.py           # Custom Stable Diffusion pipeline with geometric extensions
├── bvp_algorithm.py      # Boundary Value Problem solver for geodesic computation
├── geodesic.py           # Spherical cubic splines and geometric utilities
├── semantic_utils.py     # Pullback metric and local basis computation
├── score.py              # Score function processing
├── image_io.py           # Image input/output utilities
├── scheduler.py          # Diffusion scheduling utilities
└── args_parser.py        # Command-line argument parsing
```

## Installation

### Prerequisites
- Docker and Docker Compose
- NVIDIA Docker runtime (for GPU support)
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional but recommended)

### Docker Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/jamesballaro/geometric-diffusion.git
cd geometric-diffusion
```

#### 2. Build the Docker Image
```bash
# Build the Docker image with VFX dependencies
docker build -t geometric-diffusion -f docker/Dockerfile .
```

The Docker image includes:
- **Base**: Hugging Face Transformers PyTorch container
- **VFX Libraries**: IMath, OpenEXR, OCIO, OIIO for professional image processing
- **Python Dependencies**: All required packages from `docker/requirements.txt`
- **System Dependencies**: Build tools, development libraries, and utilities

#### 3. Run the Container
```bash
# Run with GPU support (recommended)
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/your/models:/model \
  geometric-diffusion

# Run without GPU (CPU only)
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/your/models:/model \
  geometric-diffusion
```

#### 4. Download Stable Diffusion Model
```bash
# Inside the container, download the model
# Place in /model/stable-diffusion-2-1-base/
```

### Alternative: Docker Compose (Recommended)

Run:
```bash
docker-compose up --build
```

## Usage

### Basic Example
```bash
# Inside the Docker container
cd /workspace
python src/main.py
```

### Configuration
The system supports extensive configuration through the `BVPAlgorithm` class:

```python
interpolation = BVPAlgorithm(
    # Input images and prompts
    image_path1='assets/dog17_0.png',
    image_path2='assets/dog17_1.png',
    prompt1="a cute dog",
    prompt2="a cute dog",
    
    # Geodesic optimization parameters
    bvp_opt_args={
        "opt_max_iter": 400,
        "opt_lr": 0.1,
        "lr_scheduler": 'linear',
    },
    
    # Semantic editing parameters
    semantic_edit_args={
        "image_idx": 2,
        "op": 'mid',
        "edit_prompt": "big ears",
        "x_guidance_strength": 0.3
    }
)
```

## Key Algorithms

### 1. Geodesic Computation
The BVP algorithm optimizes control points on a spherical cubic spline to minimize the geodesic distance between two latent representations:

```python
# Initialize spline with endpoints
self.spline = SphericalCubicSpline(control_points, end_points)

# Optimize through iterative refinement
for iteration in range(max_iterations):
    self.step()  # Update control points
```

### 2. Pullback Metric Analysis
Local basis vectors are discovered through power iteration on the Jacobian of the feature mapping:

```python
# Compute local encoder pullback
u, s, vT = self.unet.local_encoder_pullback_zt(
    sample, timestep, op='mid', block_idx=0
)

# Apply semantic editing along basis vectors
edited_latent = latent + alpha * basis_vector
```

## Docker Environment Details

### Included Dependencies
- **PyTorch**: Latest stable version with CUDA support
- **Diffusers**: Hugging Face diffusion models library
- **Transformers**: For text encoding and model loading
- **VFX Libraries**: Professional image processing stack
  - IMath: Mathematical utilities
  - OpenEXR: High dynamic range image format
  - OCIO: Color management
  - OIIO: Image I/O operations
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Pillow

### Volume Mounts
- **Workspace**: Current directory mounted to `/workspace`
- **Models**: Mount your model directory to `/model`
- **Results**: Output directory for generated images

### GPU Support
The Docker setup supports NVIDIA GPUs through the NVIDIA Container Toolkit:
```bash
# Verify GPU access inside container
python -c "import torch; print(torch.cuda.is_available())"
```

## Research Contributions

### From Paper 1 (Yu et al.)
- **Novel Geodesic Formulation**: Compute geodesics in diffusion latent space using probability density as metric
- **BVP Solver**: Efficient algorithm for solving boundary value problems in high-dimensional latent spaces
- **Video Analysis**: Framework for analyzing how video clips approximate geodesics in image diffusion space

### From Paper 2 (Park et al.)
- **Pullback Metric Theory**: Mathematical framework connecting latent space geometry to feature space structure
- **Local Basis Discovery**: Method for finding semantically meaningful directions in latent space
- **Single-Timestep Editing**: Efficient editing approach requiring only one manipulation step

### Integration Benefits
- **Enhanced Interpolation**: Combines smooth geodesic paths with semantic awareness
- **Controlled Generation**: Provides fine-grained control over intermediate image generation
- **Theoretical Foundation**: Bridges probability density and Riemannian geometric perspectives
- **Plug-and-play**: Neither method requires model training (although there is some prompt embedding inversion in the first algorithm). *This means the whole method can be used on a frozen/pre-trained stable diffusion network*.

## Citation

If you use this code in your research, please cite both original papers:

```bibtex
@article{yu2025probability,
  title={Probability Density Geodesics in Image Diffusion Latent Space},
  author={Yu, Qingtao and Singh, Jaskirat and Yang, Zhaoyuan and Tu, Peter Henry and Zhang, Jing and Li, Hongdong and Hartley, Richard and Campbell, Dylan},
  journal={arXiv preprint arXiv:2504.06675},
  year={2025}
}

@article{park2023understanding,
  title={Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry},
  author={Park, Yong-Hyun and Kwon, Mingi and Choi, Jaewoong and Jo, Junghyo and Uh, Youngjung},
  journal={arXiv preprint arXiv:2307.12868},
  year={2023}
}
```

<!-- ## License

This project is licensed under the MIT License - see the LICENSE file for details. -->

## Acknowledgments

- **Yu et al.** for the probability density geodesic framework
- **Park et al.** for the pullback metric analysis methodology
- **Stability AI** for the Stable Diffusion model
- **Hugging Face** for the Diffusers library