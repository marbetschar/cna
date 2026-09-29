# 3D Network Training Visualization

This directory contains scripts for creating 3D visualizations of the neural network training process from the Cooperative Network Architecture.

## Overview

The visualization shows:
- **X-axis**: Input connections (presynaptic neurons × kernel spatial dimensions)
- **Y-axis**: Output neurons (postsynaptic)
- **Z-axis**: Training epochs
- **Color**: Connection strength (weight value)

The 3D heatmap animates over training epochs, showing how the connection weights evolve as the network learns.

## Files

1. **`visualize_3d_training.py`** - Main visualization script with comprehensive features
2. **`test_3d_viz.py`** - Simplified test script for quick validation

## Quick Start

### Run a test visualization (fast)

```bash
cd src
timeout 300 python test_3d_viz.py --epochs 3 --output ../tmp/test_3d.mp4
```

This will:
- Train the network for 3 epochs with a small dataset
- Generate a 3D video showing weight evolution
- Save to `../tmp/test_3d.mp4`

### Run full visualization with more options

```bash
cd src
python visualize_3d_training.py net-fragments \
    --epochs 10 \
    --output_dir ../tmp/3d_viz/ \
    --video_name training_process.mp4 \
    --train_images 50 \
    --test_images 10 \
    --neuron_subsample 30 \
    --fps 5 \
    --dpi 150 \
    --show_2d_slices
```

**Options:**
- `config`: Configuration file name (e.g., "net-fragments" or "autoencoder")
- `--epochs`: Number of training epochs (default: 10)
- `--output_dir`: Directory to save outputs (default: "../tmp/3d_viz/")
- `--video_name`: Name of the output video file (default: "training_3d_heatmap.mp4")
- `--train_images`: Number of training images (default: from config, use smaller for faster testing)
- `--test_images`: Number of test images (default: from config)
- `--neuron_subsample`: Subsample output neurons for visualization (default: None = all)
- `--fps`: Frames per second for video (default: 5)
- `--dpi`: Video DPI (default: 150)
- `--show_2d_slices`: Also generate 2D weight matrix images at each epoch

## Output Files

The script generates:

1. **3D Video (`training_3d_heatmap.mp4`)** - Main animated 3D heatmap showing weight evolution
2. **2D Slices (optional)** - Individual weight matrix images per epoch in `2d_slices/` subdirectory
3. **Weight Distribution Plot (`weight_distribution.png`)** - Statistics of weight evolution over training:
   - Mean connection strength
   - Max connection strength
   - Number of strong connections (> 0.5)
   - Sparsity percentage
4. **Neuron Activation Heatmap (`neuron_activations.png`)** - Visualization of neuron activations

## Network Architecture

The visualization focuses on the **S2 (Lateral Layer)** of the network:
- Output channels: 40 (4 base channels × 10 alternative cells)
- Input channels: 44 (4 from feature extractor + 40 from lateral connections)
- Kernel size: 11×11 (based on locality_size=5)
- Total connections per output neuron: 44 × 11 × 11 = 5,324

## Requirements

- FFmpeg (for video encoding) - usually installed by default on Linux
- matplotlib >= 3.8.0
- numpy >= 1.24.0
- torch
- pytorch-lightning

## Performance Notes

- Training with the full dataset (300 images) takes ~6-7 seconds per image on CPU
- With 10 images per epoch and 3 epochs, expect ~3-4 minutes total
- Use `--train_images` and `--test_images` to reduce dataset size for faster testing
- Use `--neuron_subsample` to visualize a subset of neurons if the matrix is too large

## Example Commands

### Minimal test (30 seconds)
```bash
python visualize_3d_training.py net-fragments --epochs 2 --train_images 5 --output_dir ../tmp/quick_test/
```

### Standard visualization (2-3 minutes)
```bash
python visualize_3d_training.py net-fragments --epochs 3 --train_images 20 --neuron_subsample 30
```

### High-quality with all features (5-10 minutes)
```bash
python visualize_3d_training.py net-fragments \
    --epochs 5 \
    --train_images 50 \
    --neuron_subsample 40 \
    --dpi 200 \
    --fps 10 \
    --show_2d_slices
```

## Troubleshooting

### FFmpeg not found
Install FFmpeg:
- Ubuntu: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

### Out of memory
Reduce the dataset size with `--train_images` and `--test_images` parameters.

### Video too large
Use `--neuron_subsample` to reduce the number of neurons visualized, or lower `--dpi`.

### Numpy compatibility errors
The script has been tested with numpy 1.26.4. If you encounter errors with older versions, upgrade numpy:
```bash
pip install --upgrade numpy
```

## Technical Details

### Weight Extraction
The script extracts weights from the lateral layer's `W_lateral` parameter, which contains the connection strengths between neurons. These are reshaped from 4D (out_channels, in_channels, kernel_h, kernel_w) to 2D (out_neurons, input_connections) for visualization.

### Color Mapping
- Uses the `viridis` colormap
- Weights are normalized to [0, 1] range
- Global min/max are computed across all epochs for consistent color scaling

### Animation
- Uses matplotlib's `FuncAnimation` and `FFMpegWriter`
- Each frame represents one training epoch
- The Z-axis (height) of the surface corresponds to the epoch number
