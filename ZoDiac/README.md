# Attack-Resilient Image Watermarking Using Stable Diffusion (NeurIPS2024)

This is the website for reproduce results of paper "Attack-Resilient Image Watermarking Using Stable Diffusion". 
The arXiv version can be found [here](https://arxiv.org/pdf/2401.04247.pdf).

# How To Use

## Environment Setup

Prepare the conda environment by running:
```bash
conda env create -f environment.yml
conda activate my-zodiac
```

Prepare the pytorch compatible cuda 12.9 by running:
```bash
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

## Configuration

Edit `example/config/config.yaml` to set your options:

| Parameter | Description |
|-----------|-------------|
| `model_id` | HuggingFace model ID for Stable Diffusion (default: `Manojb/stable-diffusion-2-1-base`) |
| `save_img` | Output directory for watermarked images |
| `w_type` | Watermark type: `single` or `multi` |
| `w_channel` / `w_radius` / `w_seed` | Watermark pattern parameters |
| `iters` | Number of training iterations (default: 100) |
| `save_iters` | Iteration checkpoints to save (e.g., `[100]`) |
| `loss_weights` | Weights for `[L2, Watson-VGG, SSIM, WM-L1]` losses |
| `ssim_threshold` | SSIM threshold for adaptive enhancement (default: 0.92) |

## Notebook Interface

Refer to `Example.ipynb` for an interactive walkthrough. Each section can be executed separately.

## Command-Line Interface (`demo_test.py`)

Place your input image in `example/input/` and run the pipeline steps using the `--op` flag:

```bash
# Step 1: Embed watermark (100 training iterations → saves pepper_100.png)
python demo_test.py --op 1

# Step 2: Adaptive enhancement (adjusts blending to meet SSIM threshold)
python demo_test.py --op 2

# Step 3: Apply individual attacks to the watermarked image
python demo_test.py --op 3

# Step 4: Apply combined attack sequences (w/ and w/o rotation)
python demo_test.py --op 4

# Step 5: Detect watermark in original and attacked images
python demo_test.py --op 5
```

**Typical workflow:** run steps 1 → 2 → 3/4 → 5 in order.

### Supported Attackers

| Attacker | Description |
|----------|-------------|
| `diff_attacker_60` | Stable diffusion regeneration (noise step 60) |
| `cheng2020-anchor_3` | VAE compression (Cheng2020, quality 3) |
| `bmshj2018-factorized_3` | VAE compression (BMSHJ2018, quality 3) |
| `jpeg_attacker_50` | JPEG compression (quality 50) |
| `rotate_90` | 90° rotation |
| `brightness_0.5` | Brightness reduction |
| `contrast_0.5` | Contrast reduction |
| `Gaussian_noise` | Gaussian noise (std=0.05) |
| `Gaussian_blur` | Gaussian blur (kernel=5, σ=1) |
| `bm3d` | BM3D denoising |

### Output Structure

```
example/output/
├── pepper_100.png                  # watermarked image after 100 iters
├── pepper_100_SSIM0.92.png         # after adaptive enhancement
├── diff_attacker_60/
│   └── pepper_100_SSIM0.92.png
├── jpeg_attacker_50/
│   └── pepper_100_SSIM0.92.png
├── all/                            # combined attack (w/ rotation)
│   └── pepper_100_SSIM0.92.png
└── all_norot/                      # combined attack (w/o rotation)
    └── pepper_100_SSIM0.92.png
```
