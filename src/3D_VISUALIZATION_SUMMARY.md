# 3D Neural Network Structure Visualization - Summary

## Overview

I've created a **3D visualization of the neural network training process** that shows the network as a grid structure with neurons as points in 3D space and connections between them as lines with color/width representing connection strength. The visualization animates over training epochs to show how the network structure evolves in response to the input.

## What Was Created

### Main Script
**`visualize_3d_network_structure_enhanced.py`** - An enhanced visualization script that creates multiple outputs:

1. **3D Network Structure Video** (`network_structure_3d.mp4`)
   - Shows the network as a 3D grid
   - Blue points: Output neurons (40)
   - Red points: Input neurons (44)
   - Lines between neurons: Connections with:
     - **Color**: Representing connection strength (plasma colormap: purple → yellow)
     - **Width**: Proportional to connection strength
     - **Opacity**: Stronger connections are more opaque
   - Left panel: 3D network structure
   - Right top panel: Input image
   - Right bottom panel: S1 feature maps
   - Shows evolution over training epochs

2. **2D Connection Matrix Video** (`connection_matrix_2d.mp4`)
   - Shows the weight matrix (output × input neurons) evolving over time
   - Color-coded with plasma colormap
   - Helps see which connections are strengthening

3. **Input and Features Visualization** (`input_features.png`)
   - Shows the original input image
   - Shows the S1 feature maps extracted from the input
   - Helps understand what features the network is responding to

4. **Connection Statistics Plot** (`connection_stats.png`)
   - Shows weight statistics over training:
     - Average and max connection strengths
     - Number of connections above various thresholds
     - Sparsity (percentage of near-zero weights)
     - Weight distribution at the final epoch

## Example Output

Run from `/home/beso/Development/cna/src`:

```bash
python visualize_3d_network_structure_enhanced.py net-fragments \
    --epochs 3 \
    --train_images 5 \
    --output_dir ../tmp/3d_final_output/ \
    --min_connection 0.001 \
    --connection_threshold 0.05 \
    --fps 3 \
    --dpi 150
```

This creates:
- `../tmp/3d_final_output/network_structure_3d.mp4` (104KB, 1s, 2400x1800, 3 fps)
- `../tmp/3d_final_output/connection_matrix_2d.mp4` (37KB)
- `../tmp/3d_final_output/input_features.png` (28KB)
- `../tmp/3d_final_output/connection_stats.png` (182KB)

## Network Structure in the Visualization

- **Output Neurons (40)**: Arranged in a 2D grid at z=0
  - Blue color
  - Larger point size
  - Represent the S2 lateral layer neurons

- **Input Neurons (44)**: Arranged in a circle at z=-3
  - Red color
  - Smaller point size
  - Represent the input to S2 (S1 features + previous lateral outputs)

- **Connections**: Lines between output and input neurons
  - Color: Plasma colormap (purple=weak, yellow=strong)
  - Width: 0.3 to 2.5 pixels based on strength
  - Opacity: 0.3 for weak, 1.0 for strong
  - Only connections above `--min_connection` (0.001) are shown
  - Strong connections (> `--connection_threshold`) are highlighted

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs to visualize |
| `--train_images` | 2 | Number of training images per epoch |
| `--test_images` | 1 | Number of test images |
| `--output_dir` | `../tmp/3d_enhanced/` | Output directory |
| `--dpi` | 200 | Video resolution (dots per inch) |
| `--fps` | 5 | Frames per second |
| `--min_connection` | 0.001 | Minimum weight to show any connection |
| `--connection_threshold` | 0.05 | Threshold for "strong" connections |
| `--layout` | `circle` | Layout for input neurons: `circle`, `grid`, or `layers` |

## Quick Test

For a fast test (30-60 seconds):
```bash
python visualize_3d_network_structure_enhanced.py net-fragments \
    --epochs 2 \
    --train_images 1 \
    --output_dir ../tmp/3d_quick_test/
```

## What You'll See

### In the 3D Network Video:

1. **Epoch 1**: Initially, connections are very weak and sparse (mean weight ≈ 0.0004)
   - Most lines are thin and transparent
   - Only a few connections might be visible above the threshold

2. **As training progresses**: Connections start to strengthen
   - Lines become thicker and more colorful (yellow/purple)
   - Specific patterns emerge showing which input neurons connect to which output neurons
   - The network develops structure based on the input it's seeing

3. **Input Image**: Shown on the right side
   - The straight line pattern that the network is trained on
   - Helps correlate network structure with input

4. **Feature Maps**: Shown below the input image
   - Shows what S1 has extracted from the input
   - Typically shows edge/line detectors activating

### In the Connection Matrix Video:
- Heatmap showing all connections
- Rows: Output neurons
- Columns: Input neurons
- Color intensity: Connection strength
- Shows the matrix becoming more structured over time

### In the Statistics Plot:
- Tracks how weights evolve
- Shows the network becoming more sparse or more connected depending on learning

## Technical Details

- **Network**: Cooperative Network Architecture with S1 (feature extractor) and S2 (lateral connections)
- **S2 Layer**: 40 output neurons, 44 input neurons (4 from S1 + 40 from S2 feedback)
- **Total Connections**: 40 × 44 = 1,760 possible connections
- **Learning Rule**: Hebbian learning with correlation-based weight updates
- **Visualization**: Matplotlib 3D with FFmpeg video encoding

## Requirements

- Python 3.10+
- matplotlib ≥ 3.8.0
- numpy ≥ 1.26.0
- torch
- pytorch-lightning
- FFmpeg (for video encoding)

All requirements are already installed in the project's virtual environment.

## Files Created

1. **`visualize_3d_network_structure_enhanced.py`** - Main visualization script
2. **`visualize_3d_network_structure.py`** - Original version (less features)
3. **`test_3d_viz.py`** - Simplified test script
4. **`visualize_3d_training.py`** - Original 3D heatmap version

## Usage Recommendations

- For quick testing: Use `--epochs 2 --train_images 1`
- For better visualizations: Use `--epochs 5 --train_images 5` (takes ~3-5 minutes)
- For high-quality output: Use `--dpi 200 --fps 3`
- To see more connections: Lower `--connection_threshold` to 0.01 or 0.001
- To see only strong connections: Raise `--connection_threshold` to 0.1 or 0.2

## Example Output Files

The most recent run created these files in `/home/beso/Development/cna/tmp/3d_final_output/`:
- `network_structure_3d.mp4` (104KB) - Main 3D animation
- `connection_matrix_2d.mp4` (37KB) - 2D matrix evolution
- `input_features.png` (28KB) - Input image and features
- `connection_stats.png` (182KB) - Training statistics

These videos and images show the network structure evolving over 3 epochs of training with 5 images per epoch.
