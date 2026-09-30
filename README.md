# The Cooperative Network Architecture

This repository contains the code of the paper [The Cooperative Network Architecture: Learning Structured Networks as Representation of Sensory Patterns](https://arxiv.org/pdf/2407.05650).

Please note that the original code is available at [https://github.com/sagerpascal/lateral-connections](https://github.com/sagerpascal/lateral-connections), containing many more options (including a feedback layer S3 and more datasets).
However, this repository contains a more recent version of the code, which is more modular and easier to use.
All results in the paper have been re-created using this code.


## Setup
Create conda environment

```bash
conda create --name net-fragments python=3.10
```

Activate Environment

```bash
conda activate net-fragments
```

Install requirements

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run experiments

Training can be done as follows:

```bash
python main_training.py net-fragments --wandb --plot --store <store_path>
```

where `net-fragments` is the name of the training configuration, `store_path` is the path where the results should be stored, and `wandb` and `plot` are optional flags to enable logging to [wandb](https://wandb.ai/) and plotting of the results, respectively.

#### Evaluation

For evaluation, first create the baseline activations of the trained model:


```bash
python main_evaluation.py $config --load <ckpt_path> --noise 0 --line_interrupt 0 --store_baseline_activations_path <baseline_path>

```

Then run the evaluation:

```bash
python main_evaluation.py $config --load <ckpt_path> --noise <noise_float> --line_interrupt <interrupt_int> --load_baseline_activations_path <baseline_path> --act_threshold <act_threshold> --square_factor <square_factor> --wandb
```

where `noise` is the noise level, `line_interrupt` is the number of interrupted lines, `act_threshold` is the activation threshold, and `square_factor` is the square factor.
You can also use `--act_threshold bernoulli` to test with Bernoulli neurons. However, in this case, the results are based
on randomness and will vary between runs. Therefore, we recommend using a fixed activation threshold, e.g. `--act_threshold 0.5`, which will also makes the plot easier to comprehend.


---

## 3D Network Visualization

The repository includes a powerful 3D visualization tool to help understand how the network structure evolves during training.

### Quick Start

Run the main visualization from the `src/` directory:

```bash
cd src
python visualize_3d.py net-fragments \
    --epochs 3 \
    --train_images 2 \
    --output_dir ../tmp/3d_output/
```

This generates:
- `network_structure_3d.mp4` - 3D animation of network structure
- `connection_matrix_2d.mp4` - 2D connection matrix evolution
- `input_features.png` - Input image and S1 feature maps
- `connection_stats.png` - Training statistics

### Visualization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 5 | Number of training epochs to visualize |
| `--train_images` | 2 | Training images per epoch |
| `--test_images` | 1 | Test images for feature extraction |
| `--output_dir` | `../tmp/3d_enhanced/` | Output directory |
| `--dpi` | 200 | Video resolution |
| `--fps` | 5 | Frames per second |
| `--connection_threshold` | 0.05 | Threshold for "strong" connections |
| `--min_connection` | 0.001 | Minimum weight to show any connection |
| `--layout` | `circle` | Input neuron layout: `circle`, `grid`, or `layers` |

### What You See

The **3D Network Structure Video** shows:
- **Blue points**: Output neurons (S2 layer) arranged in a grid at z=0
- **Red points**: Input neurons (from S1 features) at z=-3
- **Lines**: Connections between neurons with:
  - **Color**: Connection strength (plasma colormap: purple → yellow)
  - **Width**: Proportional to connection strength
  - **Opacity**: Stronger = more opaque

Additional panels:
- **Input image**: The pattern being learned (top-right)
- **S1 feature maps**: Extracted features (bottom-right)

### Running Tests

A quick test visualization can be run from the `tests/` directory:

```bash
cd tests
python test_visualization.py --epochs 2 --output ../../tmp/test_visualization.mp4
```

This runs a minimal 2-epoch training with subsampled neurons to verify the visualization works correctly.

### Requirements

- FFmpeg (for video encoding): `sudo apt install ffmpeg` or `brew install ffmpeg`
- All Python requirements are in `requirements.txt`
