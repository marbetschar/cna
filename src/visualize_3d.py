"""
Enhanced 3D Network Visualization with Neuron Firing States and Lateral Connections

This script creates an advanced 3D visualization that shows:
1. The network as a 3D grid structure with neurons as points
2. Neuron firing states: lime green for firing, grey for inactive
3. Input connections (from feature extractor to lateral network) with strength-based opacity/width
4. Lateral connections (between neurons in the lateral network) with strength-based opacity/width
5. Animation over training epochs showing evolution
6. Evolution of lateral connections over timesteps

Connection strength is visualized with a gradient from transparent (weight 0) to black (weight 1.0).
The visualization provides a comprehensive view of how the network behaves during training.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
import torch
from tqdm import tqdm

# Import from local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import loaders_from_config
from models.s1 import FixedFilterFeatureExtractor
from models.s2_fragments import LateralNetwork
from utils.config import get_config
from lightning import Fabric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced 3D Network Visualization with Firing States and Lateral Connections')
    parser.add_argument("config", type=str, help="Config file name (e.g., net-fragments)")
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train and visualize (default: 5)')
    parser.add_argument('--output_dir', type=str, default="../tmp/3d_enhanced/",
                        help='Directory to save the visualization outputs')
    parser.add_argument('--video_name', type=str, default="network_evolution_enhanced.mp4",
                        help='Name of the output video file')
    parser.add_argument('--dpi', type=int, default=200,
                        help='DPI for the video (default: 200)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for the video (default: 5)')
    parser.add_argument('--connection_threshold', type=float, default=0.05,
                        help='Minimum weight to show as a strong connection (default: 0.05)')
    parser.add_argument('--min_connection', type=float, default=0.001,
                        help='Minimum weight to show any connection at all (default: 0.001)')
    parser.add_argument('--layout', type=str, default='circle',
                        choices=['circle', 'grid', 'layers'],
                        help='Layout for input neurons (default: circle)')
    parser.add_argument('--show_lateral_connections', type=bool, default=True,
                        help='Whether to show lateral connections (default: True)')
    parser.add_argument('--lateral_threshold', type=float, default=0.01,
                        help='Minimum lateral weight to visualize (default: 0.01)')
    return parser.parse_args()


def setup_fabric(config: Dict[str, Any]) -> Fabric:
    """Setup the fabric instance."""
    fabric = Fabric(accelerator="cpu", devices=1, loggers=[], callbacks=[])
    fabric.launch()
    fabric.seed_everything(42)
    return fabric


def setup_dataloader(config: Dict[str, Any], fabric: Fabric):
    """Setup the dataloaders."""
    train_loader, eval_loader, _ = loaders_from_config(config)
    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)
    return train_loader, eval_loader


def setup_feature_extractor(config: Dict[str, Any], fabric: Fabric):
    """Setup the feature extractor."""
    feature_extractor = FixedFilterFeatureExtractor(config, fabric)
    feature_extractor = fabric.setup(feature_extractor)
    return feature_extractor


def setup_lateral_network(config: Dict[str, Any], fabric: Fabric) -> LateralNetwork:
    """Setup the lateral network."""
    return fabric.setup(LateralNetwork(config, fabric))


def get_weights_3d(lateral_network: LateralNetwork) -> np.ndarray:
    """Extract the 4D weight matrix from the lateral network."""
    s2_layer = lateral_network.model.s2
    weights_3d = s2_layer.get_weights()
    return weights_3d.detach().cpu().numpy()


def get_lateral_weights_2d(lateral_network: LateralNetwork) -> np.ndarray:
    """Extract lateral connections as a 2D matrix (output neuron -> output neuron)."""
    s2_layer = lateral_network.model.s2
    # Get the lateral weights in their raw form
    W_lateral = s2_layer.W_lateral.data.detach().cpu().numpy()
    
    # Reshape to get connection strengths between output neurons
    # W_lateral shape: (out_channels, in_channels * kernel_h * kernel_w, 1, 1)
    # We need to extract connections that represent lateral connections
    # For simplicity, we'll look at the self-connections and nearby connections
    
    out_channels = s2_layer.out_channels
    in_channels = s2_layer.in_channels
    kernel_size = s2_layer.kernel_size
    
    # Extract the lateral part - connections from output channels back to themselves
    # This is a simplified representation showing effective lateral connectivity
    lateral_weights = W_lateral.reshape(out_channels, in_channels, kernel_size[0], kernel_size[1])
    
    # For lateral connections between output neurons, we look at the center of the kernel
    # where each output neuron connects to input from the same spatial location
    center_h, center_w = kernel_size[0] // 2, kernel_size[1] // 2
    
    # Create a 2D matrix showing effective lateral connectivity
    # This represents how strongly each output neuron is connected to others through lateral pathways
    lateral_conn_2d = np.zeros((out_channels, out_channels))
    
    for i in range(out_channels):
        for j in range(out_channels):
            # Check if output neuron i receives input from output neuron j
            # Output neurons are the last in_channels (after feature channels)
            if j < in_channels - out_channels:  # This is a feature channel, not lateral
                continue
            
            # Map input channel index to the corresponding output neuron index
            lat_idx = j - (in_channels - out_channels)
            if lat_idx < 0 or lat_idx >= out_channels:
                continue
                
            # Get the weight at the center position (most direct connection)
            weight_val = lateral_weights[i, j, center_h, center_w]
            lateral_conn_2d[i, lat_idx] = weight_val
    
    return lateral_conn_2d


def create_neuron_layout(n_neurons: int, layout: str = 'circle', radius: float = 3.0) -> np.ndarray:
    """
    Create 3D positions for neurons based on the specified layout.
    
    Args:
        n_neurons: Number of neurons
        layout: 'circle', 'grid', or 'layers'
        radius: Radius for circle layout
    
    Returns:
        positions: (n_neurons, 3) array of 3D positions
    """
    positions = np.zeros((n_neurons, 3))
    
    if layout == 'circle':
        # Arrange in a circle
        for i in range(n_neurons):
            angle = 2 * np.pi * i / n_neurons
            positions[i, 0] = radius * np.cos(angle)
            positions[i, 1] = radius * np.sin(angle)
            positions[i, 2] = 0
    
    elif layout == 'grid':
        # Arrange in a square grid
        cols = int(np.ceil(np.sqrt(n_neurons)))
        rows = int(np.ceil(n_neurons / cols))
        spacing = radius / max(cols, rows)
        
        for i in range(n_neurons):
            row = i // cols
            col = i % cols
            positions[i, 0] = col * spacing
            positions[i, 1] = row * spacing
            positions[i, 2] = 0
        
        # Center
        positions[:, 0] -= positions[:, 0].mean()
        positions[:, 1] -= positions[:, 1].mean()
    
    elif layout == 'layers':
        # Arrange in layers (for hierarchical networks)
        # Split neurons into groups
        n_groups = 4
        group_size = n_neurons // n_groups
        
        for group in range(n_groups):
            start = group * group_size
            end = start + group_size if group < n_groups - 1 else n_neurons
            
            # Each group in a circle at different z-level
            for i in range(start, end):
                idx_in_group = i - start
                angle = 2 * np.pi * idx_in_group / (end - start)
                positions[i, 0] = radius * np.cos(angle)
                positions[i, 1] = radius * np.sin(angle)
                positions[i, 2] = group * 1.0 - 1.5  # Spread along z-axis
    
    return positions


def create_network_positions(
    n_out: int,
    n_in: int,
    layout: str = 'circle'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D positions for input and output neurons.
    
    Output neurons are placed at z=0 in a grid.
    Input neurons are placed at z=-distance in the specified layout.
    Lateral connections are drawn between output neurons at the same z=0 level.
    """
    # Output neurons at z=0 in a grid
    cols_out = int(np.ceil(np.sqrt(n_out)))
    rows_out = int(np.ceil(n_out / cols_out))
    spacing_out = 1.5
    
    output_positions = np.zeros((n_out, 3))
    for i in range(n_out):
        row = i // cols_out
        col = i % cols_out
        output_positions[i, 0] = col * spacing_out
        output_positions[i, 1] = row * spacing_out
        output_positions[i, 2] = 0
    
    # Center output neurons
    output_positions[:, 0] -= output_positions[:, 0].mean()
    output_positions[:, 1] -= output_positions[:, 1].mean()
    
    # Input neurons at z=-3.0 in the specified layout
    input_positions = create_neuron_layout(n_in, layout, radius=2.5)
    input_positions[:, 2] -= 3.0  # Move behind output neurons
    
    return output_positions, input_positions


def extract_all_connections(weights_4d: np.ndarray) -> np.ndarray:
    """
    Extract all connection weights as a 2D matrix.
    
    Args:
        weights_4d: (out_channels, in_channels, kernel_h, kernel_w)
    
    Returns:
        weights_2d: (out_channels, in_channels) matrix of mean weights
    """
    return weights_4d.mean(axis=(2, 3))


def run_training_and_capture(
    config: Dict[str, Any],
    feature_extractor,
    lateral_network: LateralNetwork,
    train_loader,
    test_loader,
    epochs: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Run training and capture network state at each epoch.
    
    Returns:
        weight_history: List of 2D weight matrices (out, in)
        activation_history: List of neuron activation states (firing or not)
        lateral_weight_history: List of lateral connection matrices
        input_image: Sample input image
        input_features: Sample input features
    """
    feature_extractor.eval()
    lateral_network.eval()
    
    weight_history = []
    activation_history = []  # Will store firing states for each epoch
    lateral_weight_history = []  # Will store lateral connection evolution
    input_image = None
    input_features = None
    
    # Get a sample input
    for batch in test_loader:
        sample_batch = batch[0]
        with torch.no_grad():
            input_image = sample_batch.clone()
            input_features = feature_extractor(sample_batch)
        break
    
    # Run training epochs
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}...")
        
        epoch_activations = []  # Store activations across timesteps
        
        # Training loop
        for i, batch in enumerate(tqdm(train_loader, 
                                        total=len(train_loader),
                                        colour="GREEN",
                                        desc=f"  Training")):
            with torch.no_grad():
                batch_features = feature_extractor(batch[0])
            
            z = None
            for view_idx in range(batch_features.shape[1]):
                x_view_features = batch_features[:, view_idx, ...]
                
                if z is None:
                    z = torch.zeros((x_view_features.shape[0], lateral_network.model.out_channels, 
                                     x_view_features.shape[2], x_view_features.shape[3]), 
                                    device=batch[0].device)
                
                features_lat = []
                timestep_activations = []  # Store activations for each timestep
                
                for t in range(config["lateral_model"]["max_timesteps"]):
                    lateral_network.model.update_ts(t)
                    x_in = torch.cat([x_view_features, z], dim=1)
                    
                    # Get activations (firing states) at this timestep
                    with torch.no_grad():
                        _, z_bin, _ = lateral_network.model.s2(x_in)
                        # z_bin contains the binary firing states
                        # Flatten spatial dimensions to get per-neuron firing
                        firing_states = z_bin.mean(dim=(2, 3))  # Average across spatial positions
                        timestep_activations.append(firing_states.cpu().numpy())
                    
                    # Continue with normal training
                    z_float, z = lateral_network(x_in)
                    features_lat.append(z)
                
                # Store the last timestep's activations for this batch
                if timestep_activations:
                    epoch_activations.append(timestep_activations[-1])
                
                features_lat = torch.stack(features_lat, dim=1)
                features_lat_median = torch.median(features_lat, dim=1)[0]
                
                # Train
                x_rearranged = lateral_network.model.s2.rearrange_input(
                    torch.cat([x_view_features, features_lat_median], dim=1))
                lateral_network.model.s2.hebbian_update(x_rearranged, features_lat_median)
        
        # Get average activations across the epoch
        if epoch_activations:
            avg_activations = np.mean(np.concatenate(epoch_activations, axis=0), axis=0)
            # Binarize: 1 if firing (activation > 0.5), 0 otherwise
            firing_states = (avg_activations > 0.5).astype(np.float32)
            activation_history.append(firing_states)
        else:
            # Default to all inactive
            firing_states = np.zeros(lateral_network.model.out_channels)
            activation_history.append(firing_states)
        
        # Capture weights
        weights_4d = get_weights_3d(lateral_network)
        weights_2d = extract_all_connections(weights_4d)
        weight_history.append(weights_2d.copy())
        
        # Capture lateral weights
        lateral_weights_2d = get_lateral_weights_2d(lateral_network)
        lateral_weight_history.append(lateral_weights_2d.copy())
        
        print(f"  Weights: min={weights_2d.min():.6f}, max={weights_2d.max():.6f}, mean={weights_2d.mean():.6f}")
        print(f"  Lateral Weights: min={lateral_weights_2d.min():.6f}, max={lateral_weights_2d.max():.6f}")
        n_strong = np.sum(weights_2d > 0.1)
        print(f"  Strong connections (>0.1): {n_strong}/{weights_2d.size}")
        n_active = np.sum(firing_states > 0)
        print(f"  Active neurons: {n_active}/{firing_states.size}")
    
    return weight_history, activation_history, lateral_weight_history, input_image, input_features


def create_enhanced_visualization(
    weight_history: List[np.ndarray],
    activation_history: List[np.ndarray],
    lateral_weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    input_positions: np.ndarray,
    input_image: Optional[torch.Tensor] = None,
    input_features: Optional[torch.Tensor] = None,
    output_dir: Path = None,
    fps: int = 5,
    dpi: int = 200,
    min_connection: float = 0.001,
    connection_threshold: float = 0.05,
    lateral_threshold: float = 0.01,
    layout: str = 'circle'
):
    """
    Create the enhanced visualization with multiple views.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Creating Enhanced 3D Network Visualization with Firing States")
    print("="*70)
    
    # 1. Main 3D network structure video with firing states
    print("\n1. Creating main 3D network structure video with firing states...")
    create_3d_network_video_with_firing(
        weight_history,
        activation_history,
        lateral_weight_history,
        output_positions,
        input_positions,
        input_image=input_image,
        input_features=input_features,
        output_path=output_dir / "network_structure_3d_firing.mp4",
        fps=fps,
        dpi=dpi,
        min_connection=min_connection,
        connection_threshold=connection_threshold,
        lateral_threshold=lateral_threshold
    )
    
    # 2. Lateral connections only video
    print("\n2. Creating lateral connections evolution video...")
    create_lateral_connections_video(
        lateral_weight_history,
        output_positions,
        output_path=output_dir / "lateral_connections.mp4",
        fps=fps,
        dpi=dpi,
        threshold=lateral_threshold
    )
    
    # 3. 2D connection matrix video
    print("\n3. Creating 2D connection matrix video...")
    create_2d_matrix_video(
        weight_history,
        output_dir / "connection_matrix_2d.mp4",
        fps=fps,
        dpi=dpi
    )
    
    # 4. Neuron activation heatmap video
    print("\n4. Creating neuron activation heatmap video...")
    create_activation_heatmap_video(
        activation_history,
        output_dir / "neuron_activations.mp4",
        fps=fps,
        dpi=dpi
    )
    
    # 5. Input image and features
    if input_image is not None:
        print("\n5. Saving input image and features...")
        save_input_visualization(
            input_image,
            input_features,
            output_dir / "input_features.png",
            dpi=dpi
        )
    
    # 6. Connection statistics
    print("\n6. Creating connection statistics...")
    create_connection_stats(
        weight_history,
        activation_history,
        lateral_weight_history,
        output_dir / "connection_stats.png",
        dpi=dpi
    )
    
    print("\n" + "="*70)
    print("All enhanced visualizations completed!")
    print(f"Output directory: {output_dir}")
    print("="*70)


def create_3d_network_video_with_firing(
    weight_history: List[np.ndarray],
    activation_history: List[np.ndarray],
    lateral_weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    input_positions: np.ndarray,
    input_image: Optional[torch.Tensor] = None,
    input_features: Optional[torch.Tensor] = None,
    output_path: Path = None,
    fps: int = 5,
    dpi: int = 200,
    min_connection: float = 0.001,
    connection_threshold: float = 0.05,
    lateral_threshold: float = 0.01
):
    """
    Create the main 3D network structure video with firing states and lateral connections.
    
    The video shows:
    - 3D grid with output neurons (green when firing, grey when inactive)
    - Input neurons (red points)
    - Input connections as lines with color/width representing strength
    - Lateral connections between output neurons
    - Input image on the side
    - Feature maps from S1
    """
    print(f"  Creating video with {len(output_positions)} output and {len(input_positions)} input neurons")
    print(f"  Frames: {len(weight_history)}")
    
    if output_path is None:
        output_path = Path("network_structure_3d_firing.mp4")
    
    # Get dimensions
    n_out = len(output_positions)
    n_in = len(input_positions)
    n_frames = len(weight_history)
    
    # Find global min/max weights across all frames
    all_weights = np.concatenate([w.flatten() for w in weight_history])
    global_min = all_weights.min()
    global_max = all_weights.max()
    
    # Find global lateral weights range
    all_lateral_weights = np.concatenate([w.flatten() for w in lateral_weight_history])
    lateral_min = all_lateral_weights.min()
    lateral_max = all_lateral_weights.max()
    
    # Create figure with 3 subplots:
    # Left: 3D network structure
    # Top-right: Input image
    # Bottom-right: Info panel
    # Reduce figure size for video encoding compatibility
    video_dpi = min(dpi, 150)  # Cap DPI for video
    fig_size = (16, 12) if dpi > 150 else (18, 14)
    fig = plt.figure(figsize=fig_size, dpi=video_dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1])
    
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_input = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])
    
    # Set up 3D view
    ax_3d.view_init(elev=20, azim=-45)
    ax_3d.set_xlabel('X', fontsize=12, labelpad=10)
    ax_3d.set_ylabel('Y', fontsize=12, labelpad=10)
    ax_3d.set_zlabel('Z', fontsize=12, labelpad=10)
    
    # Set axis limits to accommodate all neurons
    all_positions = np.vstack([output_positions, input_positions])
    x_min, x_max = all_positions[:, 0].min() - 2, all_positions[:, 0].max() + 2
    y_min, y_max = all_positions[:, 1].min() - 2, all_positions[:, 1].max() + 2
    z_min, z_max = all_positions[:, 2].min() - 1, all_positions[:, 2].max() + 1
    
    ax_3d.set_xlim([x_min, x_max])
    ax_3d.set_ylim([y_min, y_max])
    ax_3d.set_zlim([z_min, z_max])
    
    # Plot output neurons (will change color based on firing state)
    # Start with all grey (inactive)
    firing_states = activation_history[0] if activation_history else np.zeros(n_out)
    neuron_colors = np.array(['grey' if firing_states[i] < 0.5 else 'limegreen' for i in range(n_out)])
    
    output_scatter = ax_3d.scatter(
        output_positions[:, 0],
        output_positions[:, 1],
        output_positions[:, 2],
        c=neuron_colors,
        s=100,
        alpha=0.9,
        depthshade=True,
        label='Output Neurons',
        edgecolors='black',
        linewidths=0.5
    )
    
    # Plot input neurons (red points)
    input_scatter = ax_3d.scatter(
        input_positions[:, 0],
        input_positions[:, 1],
        input_positions[:, 2],
        c='red',
        s=60,
        alpha=0.9,
        depthshade=True,
        label='Input Neurons',
        edgecolors='black',
        linewidths=0.3
    )
    
    # Create all possible input connection lines
    input_lines = []
    for i in range(n_out):
        for j in range(n_in):
            line = ax_3d.plot(
                [output_positions[i, 0], input_positions[j, 0]],
                [output_positions[i, 1], input_positions[j, 1]],
                [output_positions[i, 2], input_positions[j, 2]],
                'k-',
                linewidth=0.05,
                alpha=0.0,
                zorder=1
            )[0]
            input_lines.append(line)
    
    # Create lateral connection lines (between output neurons)
    lateral_lines = []
    for i in range(n_out):
        for j in range(n_out):
            if i == j:  # Skip self-connections for clarity
                continue
            line = ax_3d.plot(
                [output_positions[i, 0], output_positions[j, 0]],
                [output_positions[i, 1], output_positions[j, 1]],
                [output_positions[i, 2], output_positions[j, 2]],
                'b-',
                linewidth=0.05,
                alpha=0.0,
                zorder=1,
                linestyle='--'
            )[0]
            lateral_lines.append(line)
    
    # Show input image
    if input_image is not None:
        img_array = input_image[0, 0].cpu().numpy() if input_image.is_cuda else input_image[0, 0].numpy()
        img_input = ax_input.imshow(img_array, cmap='gray', interpolation='nearest', aspect='auto')
        ax_input.set_title('Input Image', fontsize=12)
        ax_input.axis('off')
    else:
        ax_input.axis('off')
    
    # Info panel
    ax_info.axis('off')
    
    # Set initial title
    ax_3d.set_title(f'3D Network Structure with Firing States - Epoch 1', fontsize=14)
    
    # Add colorbar for connection strength (transparent to black)
    # Create a custom gradient from transparent to black
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad = np.vstack((grad, grad, grad, np.ones_like(grad)))
    grad = grad.reshape(-1, 4)
    custom_cmap = LinearSegmentedColormap.from_list('trans_to_black', ['white', 'black'], N=256)
    dummy_mappable = cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=0, vmax=1))
    dummy_mappable.set_array([])
    cbar = fig.colorbar(dummy_mappable, ax=ax_3d, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('Connection Strength (transparent to black)', fontsize=12)
    
    # Add custom legend for firing states
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='limegreen', edgecolor='black', label='Firing Neuron'),
        Patch(facecolor='grey', edgecolor='black', label='Inactive Neuron'),
        Patch(facecolor='red', edgecolor='black', label='Input Neuron')
    ]
    ax_3d.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    
    def update(frame: int):
        """Update function for animation."""
        weights_2d = weight_history[frame]
        firing_states = activation_history[frame] if frame < len(activation_history) else activation_history[-1]
        lateral_weights_2d = lateral_weight_history[frame] if frame < len(lateral_weight_history) else lateral_weight_history[-1]
        
        # Normalize current weights
        current_min = weights_2d.min()
        current_max = weights_2d.max()
        
        if current_max - current_min > 1e-10:
            weights_norm = (weights_2d - current_min) / (current_max - current_min)
        else:
            weights_norm = np.zeros_like(weights_2d)
        
        # Update neuron colors based on firing state
        neuron_colors = np.array(['grey' if firing_states[i] < 0.5 else 'limegreen' for i in range(n_out)])
        output_scatter.set_color(neuron_colors)
        
        # Update all input connection lines
        line_idx = 0
        max_linewidth = 2.5
        for i in range(n_out):
            for j in range(n_in):
                weight = weights_2d[i, j]
                
                if line_idx < len(input_lines):
                    if weight >= min_connection:
                        # Normalize weight for width and opacity
                        if current_max - current_min > 1e-10:
                            norm_weight = (weight - current_min) / (current_max - current_min)
                        else:
                            norm_weight = 0
                        
                        # Connection strength from transparent to black
                        # Use black color with alpha based on weight
                        alpha_val = norm_weight
                        linewidth = 0.3 + norm_weight * max_linewidth
                        
                        input_lines[line_idx].set_color('black')
                        input_lines[line_idx].set_linewidth(linewidth)
                        input_lines[line_idx].set_alpha(alpha_val)
                        input_lines[line_idx].set_zorder(2)
                    else:
                        input_lines[line_idx].set_color('gray')
                        input_lines[line_idx].set_linewidth(0.05)
                        input_lines[line_idx].set_alpha(0.05)
                        input_lines[line_idx].set_zorder(1)
                line_idx += 1
        
        # Update lateral connection lines
        lateral_line_idx = 0
        lateral_max_linewidth = 1.5
        lateral_current_max = lateral_weights_2d.max()
        lateral_current_min = lateral_weights_2d.min()
        
        for i in range(n_out):
            for j in range(n_out):
                if i == j:
                    continue
                    
                lateral_weight = lateral_weights_2d[i, j]
                
                if lateral_line_idx < len(lateral_lines):
                    if lateral_weight >= lateral_threshold:
                        # Normalize lateral weight
                        if lateral_current_max - lateral_current_min > 1e-10:
                            norm_lateral = (lateral_weight - lateral_current_min) / (lateral_current_max - lateral_current_min)
                        else:
                            norm_lateral = 0
                        
                        # Connection strength from transparent to black
                        # Use black color with alpha based on weight
                        alpha_val = norm_lateral
                        linewidth = 0.2 + norm_lateral * lateral_max_linewidth
                        
                        lateral_lines[lateral_line_idx].set_color('black')
                        lateral_lines[lateral_line_idx].set_linewidth(linewidth)
                        lateral_lines[lateral_line_idx].set_alpha(alpha_val)
                        lateral_lines[lateral_line_idx].set_zorder(3)  # Above input connections
                    else:
                        lateral_lines[lateral_line_idx].set_color('lightblue')
                        lateral_lines[lateral_line_idx].set_linewidth(0.05)
                        lateral_lines[lateral_line_idx].set_alpha(0.1)
                        lateral_lines[lateral_line_idx].set_zorder(1)
                lateral_line_idx += 1
        
        # Update title
        n_strong = np.sum(weights_2d > connection_threshold)
        n_weak = np.sum((weights_2d > min_connection) & (weights_2d <= connection_threshold))
        n_active = np.sum(firing_states > 0.5)
        n_lateral_active = np.sum(lateral_weights_2d > lateral_threshold)
        
        ax_3d.set_title(
            f'3D Network - Epoch {frame + 1}\n' \
            f'Input: Strong={n_strong}, Weak={n_weak}, Active Neurons={n_active}/{n_out}\n' \
            f'Lateral: {n_lateral_active} active connections, Weights: [{current_min:.4f}, {current_max:.4f}]', 
            fontsize=12
        )
        
        # Update info panel
        ax_info.clear()
        info_text = f"Epoch: {frame + 1}/{n_frames}\n\n" \
                   f"Input Connections:\n" \
                   f"  Strong (>0.05): {n_strong}\n" \
                   f"  Weak: {n_weak}\n" \
                   f"  Weight Range: [{current_min:.4f}, {current_max:.4f}]\n\n" \
                   f"Neuron Activity:\n" \
                   f"  Firing: {n_active}/{n_out}\n" \
                   f"  Inactive: {n_out - n_active}/{n_out}\n\n" \
                   f"Lateral Connections:\n" \
                   f"  Active (>0.01): {n_lateral_active}\n" \
                   f"  Range: [{lateral_current_min:.4f}, {lateral_current_max:.4f}]"
        
        ax_info.text(0.1, 0.9, info_text, 
                    ha='left', va='top', fontsize=11,
                    family='monospace', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))
        ax_info.set_title(f'Network Statistics - Epoch {frame + 1}', fontsize=12)
        ax_info.axis('off')
        
        # Update feature display with epoch info
        if input_features is not None and frame == 0:
            feat_array = input_features[0, :, 0].cpu().numpy() if input_features.is_cuda else input_features[0, :, 0].numpy()
            n_feats = min(4, feat_array.shape[0])
            
            # Normalize each feature map
            feat_norm = (feat_array - feat_array.min()) / (feat_array.max() - feat_array.min() + 1e-10)
            
            # Display as a grid
            grid_size = int(np.ceil(np.sqrt(n_feats)))
            ax_features = ax_input if input_image is None else ax_input
            for idx in range(n_feats):
                row = idx // grid_size
                col = idx % grid_size
                ax_features.imshow(
                    feat_norm[idx],
                    cmap='viridis',
                    interpolation='nearest',
                    extent=[col, col + 1, row, row + 1],
                    aspect='auto'
                )
            
            ax_features.set_title('S1 Feature Maps', fontsize=12)
            ax_features.axis('off')
        
        return input_lines + lateral_lines + [output_scatter, input_scatter, ax_3d]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames, 
                       interval=1000/fps, blit=False, repeat=False)
    
    # Save as video
    writer = FFMpegWriter(fps=fps, bitrate=30000, metadata={'title': '3D Network Structure with Firing States'})
    ani.save(str(output_path), writer=writer)
    
    plt.close(fig)
    print(f"  Video saved to: {output_path}")
    
    return output_path


def create_lateral_connections_video(
    lateral_weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    output_path: Path,
    fps: int = 5,
    dpi: int = 200,
    threshold: float = 0.01
):
    """Create a dedicated video showing only lateral connections evolution."""
    n_frames = len(lateral_weight_history)
    n_out = len(output_positions)
    
    print(f"  Creating lateral connections video with {n_out} neurons and {n_frames} frames")
    
    # Find global lateral weights range
    all_lateral_weights = np.concatenate([w.flatten() for w in lateral_weight_history])
    global_min = all_lateral_weights.min()
    global_max = all_lateral_weights.max()
    
    video_dpi = min(dpi, 150)
    fig = plt.figure(figsize=(12, 10), dpi=video_dpi)
    
    # Create custom colormap for transparent to black gradient
    custom_cmap = LinearSegmentedColormap.from_list('trans_to_black', ['white', 'black'], N=256)
    
    # Create 3D subplot
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.view_init(elev=25, azim=-45)
    
    # Set up axis
    ax_3d.set_xlabel('X', fontsize=12, labelpad=10)
    ax_3d.set_ylabel('Y', fontsize=12, labelpad=10)
    ax_3d.set_zlabel('Z', fontsize=12, labelpad=10)
    
    # Set limits
    x_min, x_max = output_positions[:, 0].min() - 1, output_positions[:, 0].max() + 1
    y_min, y_max = output_positions[:, 1].min() - 1, output_positions[:, 1].max() + 1
    z_min, z_max = output_positions[:, 2].min() - 1, output_positions[:, 2].max() + 1
    
    ax_3d.set_xlim([x_min, x_max])
    ax_3d.set_ylim([y_min, y_max])
    ax_3d.set_zlim([z_min, z_max])
    
    # Plot neurons as reference points
    neuron_scatter = ax_3d.scatter(
        output_positions[:, 0],
        output_positions[:, 1],
        output_positions[:, 2],
        c='blue',
        s=80,
        alpha=0.8,
        depthshade=True,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Create lateral connection lines
    lines = []
    line_indices = []  # Store (i,j) for each line
    
    for i in range(n_out):
        for j in range(n_out):
            if i == j:  # Skip self-connections
                continue
            line = ax_3d.plot(
                [output_positions[i, 0], output_positions[j, 0]],
                [output_positions[i, 1], output_positions[j, 1]],
                [output_positions[i, 2], output_positions[j, 2]],
                'b-',
                linewidth=0.05,
                alpha=0.0,
                zorder=1,
                linestyle='--'
            )[0]
            lines.append(line)
            line_indices.append((i, j))
    
    # Add colorbar for lateral connection strength (transparent to black)
    dummy_mappable = cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=global_min, vmax=global_max))
    dummy_mappable.set_array([])
    cbar = fig.colorbar(dummy_mappable, ax=ax_3d, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('Lateral Connection Strength (transparent to black)', fontsize=12)
    
    ax_3d.set_title(f'Lateral Connections Evolution - Epoch 1', fontsize=14)
    
    fig.tight_layout()
    
    def update(frame: int):
        """Update function for animation."""
        lateral_weights_2d = lateral_weight_history[frame]
        
        # Normalize current weights
        current_min = lateral_weights_2d.min()
        current_max = lateral_weights_2d.max()
        
        if current_max - current_min > 1e-10:
            weights_norm = (lateral_weights_2d - current_min) / (current_max - current_min)
        else:
            weights_norm = np.zeros_like(lateral_weights_2d)
        
        # Update all lateral connection lines
        max_linewidth = 2.0
        for idx, (i, j) in enumerate(line_indices):
            if idx < len(lines):
                lateral_weight = lateral_weights_2d[i, j]
                
                if lateral_weight >= threshold:
                    # Normalize weight
                    if current_max - current_min > 1e-10:
                        norm_weight = (lateral_weight - current_min) / (current_max - current_min)
                    else:
                        norm_weight = 0
                    
                    # Connection strength from transparent to black
                    # Use black color with alpha based on weight
                    alpha_val = norm_weight
                    linewidth = 0.3 + norm_weight * max_linewidth
                    
                    lines[idx].set_color('black')
                    lines[idx].set_linewidth(linewidth)
                    lines[idx].set_alpha(alpha_val)
                    lines[idx].set_zorder(2)
                else:
                    lines[idx].set_color('lightgrey')
                    lines[idx].set_linewidth(0.05)
                    lines[idx].set_alpha(0.1)
                    lines[idx].set_zorder(1)
        
        # Update title
        n_active = np.sum(lateral_weights_2d > threshold)
        ax_3d.set_title(
            f'Lateral Connections - Epoch {frame + 1}\n' \
            f'Active connections: {n_active}/{len(lines)}, Range: [{current_min:.4f}, {current_max:.4f}]', 
            fontsize=14
        )
        
        return lines + [neuron_scatter, ax_3d]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames, 
                       interval=1000/fps, blit=False, repeat=False)
    
    # Save as video
    writer = FFMpegWriter(fps=fps, bitrate=20000, metadata={'title': 'Lateral Connections Evolution'})
    ani.save(str(output_path), writer=writer)
    
    plt.close(fig)
    print(f"  Lateral connections video saved to: {output_path}")


def create_2d_matrix_video(
    weight_history: List[np.ndarray],
    output_path: Path,
    fps: int = 5,
    dpi: int = 200
):
    """Create a 2D video showing the connection matrix evolution."""
    matrices_array = np.stack(weight_history, axis=0)
    n_frames, n_out, n_in = matrices_array.shape
    
    global_min = matrices_array.min()
    global_max = matrices_array.max()
    
    video_dpi = min(dpi, 150)
    fig = plt.figure(figsize=(12, 10), dpi=video_dpi)
    ax = fig.add_subplot(111)
    
    img = ax.imshow(
        matrices_array[0],
        cmap='Greys',
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest',
        aspect='auto'
    )
    
    ax.set_xlabel('Input Neurons', fontsize=14)
    ax.set_ylabel('Output Neurons', fontsize=14)
    ax.set_title(f'Input Connection Matrix - Epoch 1', fontsize=16)
    
    fig.colorbar(img, ax=ax, shrink=0.7)
    fig.tight_layout()
    
    def update(frame):
        img.set_array(matrices_array[frame])
        current_min = matrices_array[frame].min()
        current_max = matrices_array[frame].max()
        n_strong = np.sum(matrices_array[frame] > 0.1)
        ax.set_title(f'Input Connection Matrix - Epoch {frame + 1}\nRange: [{current_min:.4f}, {current_max:.4f}], Strong: {n_strong}', 
                    fontsize=14)
        return img, ax
    
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    writer = FFMpegWriter(fps=fps, bitrate=15000)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    
    print(f"  2D matrix video saved to: {output_path}")


def create_activation_heatmap_video(
    activation_history: List[np.ndarray],
    output_path: Path,
    fps: int = 5,
    dpi: int = 200
):
    """Create a video showing neuron activation heatmap over epochs."""
    activations_array = np.stack(activation_history, axis=0)
    n_frames, n_neurons = activations_array.shape
    
    video_dpi = min(dpi, 150)
    fig = plt.figure(figsize=(12, 8), dpi=video_dpi)
    
    # Reshape to 2D grid for visualization
    grid_size = int(np.ceil(np.sqrt(n_neurons)))
    padded_neurons = grid_size * grid_size
    
    # Create a grid representation
    activation_grid = np.zeros((n_frames, grid_size, grid_size))
    
    for frame in range(n_frames):
        for i in range(n_neurons):
            row = i // grid_size
            col = i % grid_size
            activation_grid[frame, row, col] = activations_array[frame, i]
    
    img = plt.imshow(
        activation_grid[0],
        cmap='Greens',
        vmin=0,
        vmax=1,
        interpolation='nearest',
        aspect='auto'
    )
    
    plt.xlabel('Neuron Index (X)', fontsize=14)
    plt.ylabel('Neuron Index (Y)', fontsize=14)
    plt.title(f'Neuron Activation Heatmap - Epoch 1', fontsize=16)
    
    cbar = plt.colorbar(img, shrink=0.7)
    cbar.set_label('Activation (1 = Firing)', fontsize=12)
    
    plt.tight_layout()
    
    def update(frame):
        img.set_array(activation_grid[frame])
        n_active = np.sum(activations_array[frame] > 0.5)
        plt.title(f'Neuron Activation Heatmap - Epoch {frame + 1}\nActive: {n_active}/{n_neurons}', 
                 fontsize=14)
        return img,
    
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    writer = FFMpegWriter(fps=fps, bitrate=10000)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    
    print(f"  Activation heatmap video saved to: {output_path}")


def save_input_visualization(
    input_image: torch.Tensor,
    input_features: torch.Tensor,
    output_path: Path,
    dpi: int = 200
):
    """Save input image and feature maps."""
    video_dpi = min(dpi, 150)
    fig = plt.figure(figsize=(14, 6), dpi=video_dpi)
    
    # Input image
    ax1 = fig.add_subplot(1, 2, 1)
    img_array = input_image[0, 0].cpu().numpy() if input_image.is_cuda else input_image[0, 0].numpy()
    ax1.imshow(img_array, cmap='gray', interpolation='nearest')
    ax1.set_title('Input Image', fontsize=14)
    ax1.axis('off')
    
    # Feature maps
    ax2 = fig.add_subplot(1, 2, 2)
    feat_array = input_features[0, :, 0].cpu().numpy() if input_features.is_cuda else input_features[0, :, 0].numpy()
    
    n_feats = feat_array.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_feats)))
    
    # Normalize each feature map
    feat_norm = np.zeros_like(feat_array)
    for i in range(n_feats):
        f_min = feat_array[i].min()
        f_max = feat_array[i].max()
        if f_max - f_min > 1e-10:
            feat_norm[i] = (feat_array[i] - f_min) / (f_max - f_min)
    
    # Create grid of feature maps
    for idx in range(n_feats):
        row = idx // grid_size
        col = idx % grid_size
        ax2.imshow(
            feat_norm[idx],
            cmap='viridis',
            interpolation='nearest',
            extent=[col, col + 1, row, row + 1],
            aspect='auto'
        )
    
    ax2.set_title(f'S1 Feature Maps ({n_feats} channels)', fontsize=14)
    ax2.axis('off')
    
    fig.suptitle('Input and Extracted Features', fontsize=16)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Input visualization saved to: {output_path}")


def create_connection_stats(
    weight_history: List[np.ndarray],
    activation_history: List[np.ndarray],
    lateral_weight_history: List[np.ndarray],
    output_path: Path,
    dpi: int = 200
):
    """Create a plot showing connection statistics over training."""
    n_frames = len(weight_history)
    
    # Compute statistics
    avg_weights = [w.mean() for w in weight_history]
    max_weights = [w.max() for w in weight_history]
    
    # Number of connections above thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    n_connections = {}
    for t in thresholds:
        n_connections[t] = [np.sum(w > t) for w in weight_history]
    
    # Sparsity
    sparsity = [np.mean(w < 0.001) * 100 for w in weight_history]
    
    # Lateral connection statistics
    avg_lateral_weights = [w.mean() for w in lateral_weight_history]
    max_lateral_weights = [w.max() for w in lateral_weight_history]
    lateral_sparsity = [np.mean(w < 0.001) * 100 for w in lateral_weight_history]
    
    # Neuron activity statistics
    avg_activation = [a.mean() for a in activation_history]
    n_active_neurons = [np.sum(a > 0.5) for a in activation_history]
    
    video_dpi = min(dpi, 150)
    fig = plt.figure(figsize=(16, 12), dpi=video_dpi)
    
    # Plot 1: Average and max weights
    ax1 = fig.add_subplot(3, 3, 1)
    epochs = range(1, n_frames + 1)
    ax1.plot(epochs, avg_weights, 'b-', linewidth=2, label='Average Input Weight')
    ax1.plot(epochs, max_weights, 'r-', linewidth=2, label='Max Input Weight')
    ax1.plot(epochs, avg_lateral_weights, 'c-', linewidth=2, label='Average Lateral Weight')
    ax1.plot(epochs, max_lateral_weights, 'm-', linewidth=2, label='Max Lateral Weight')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weight Value', fontsize=12)
    ax1.set_title('Weight Statistics Over Training', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of input connections
    ax2 = fig.add_subplot(3, 3, 2)
    colors = ['g', 'c', 'b', 'm', 'r']
    for idx, t in enumerate(thresholds):
        ax2.plot(epochs, n_connections[t], f'{colors[idx]}-', linewidth=2, label=f'> {t}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Input Connections Above Threshold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Input sparsity
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(epochs, sparsity, 'm-', linewidth=2, label='Input Sparsity')
    ax3.plot(epochs, lateral_sparsity, 'c-', linewidth=2, label='Lateral Sparsity')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Sparsity (Weights < 0.001)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Neuron activation statistics
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(epochs, avg_activation, 'g-', linewidth=2, label='Avg Activation')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Activation Rate', fontsize=12)
    ax4.set_title('Average Neuron Activation', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Number of active neurons
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(epochs, n_active_neurons, 'b-', linewidth=2, label='Active Neurons')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title('Number of Firing Neurons', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Weight distribution at last epoch
    ax6 = fig.add_subplot(3, 3, 6)
    last_weights = weight_history[-1].flatten()
    ax6.hist(last_weights[last_weights > 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black', range=(0, 1))
    ax6.set_xlabel('Weight Value', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title(f'Input Weight Distribution (Epoch {n_frames})', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Lateral weight distribution at last epoch
    ax7 = fig.add_subplot(3, 3, 7)
    last_lateral_weights = lateral_weight_history[-1].flatten()
    ax7.hist(last_lateral_weights[last_lateral_weights > 0], bins=50, alpha=0.7, color='mediumpurple', edgecolor='black', range=(0, 1))
    ax7.set_xlabel('Weight Value', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.set_title(f'Lateral Weight Distribution (Epoch {n_frames})', fontsize=14)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Activation distribution
    ax8 = fig.add_subplot(3, 3, 8)
    last_activations = activation_history[-1]
    unique, counts = np.unique(last_activations, return_counts=True)
    ax8.bar(unique, counts, alpha=0.7, color='limegreen', edgecolor='black')
    ax8.set_xlabel('Activation State', fontsize=12)
    ax8.set_ylabel('Count', fontsize=12)
    ax8.set_title(f'Neuron Activation Distribution (Epoch {n_frames})', fontsize=14)
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels(['Inactive', 'Firing'])
    ax8.grid(True, alpha=0.3)
    
    # Hide unused subplot
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    fig.suptitle('Network Connection and Activation Statistics', fontsize=16)
    fig.tight_layout()
    
    fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Connection stats saved to: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    config = get_config(args.config)
    config['run']['n_epochs'] = args.epochs
    
    print("="*70)
    print("Enhanced 3D Network Visualization with Firing States and Lateral Connections")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Connection threshold: {args.connection_threshold}")
    print(f"Min connection: {args.min_connection}")
    print(f"Lateral threshold: {args.lateral_threshold}")
    print(f"Layout: {args.layout}")
    
    # Setup
    fabric = setup_fabric(config)
    train_loader, test_loader = setup_dataloader(config, fabric)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)
    
    # Get network dimensions
    s2_layer = lateral_network.model.s2
    n_out = s2_layer.out_channels
    n_in = s2_layer.in_channels
    
    print(f"\nNetwork Architecture:")
    print(f"  Output neurons: {n_out}")
    print(f"  Input neurons: {n_in}")
    print(f"  Total possible input connections: {n_out * n_in}")
    print(f"  Total possible lateral connections: {n_out * n_out}")
    
    # Create neuron positions
    output_positions, input_positions = create_network_positions(
        n_out, n_in, layout=args.layout
    )
    
    # Run training and capture state
    print("\nRunning training and capturing network state...")
    weight_history, activation_history, lateral_weight_history, input_image, input_features = run_training_and_capture(
        config,
        feature_extractor,
        lateral_network,
        train_loader,
        test_loader,
        epochs=args.epochs
    )
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    create_enhanced_visualization(
        weight_history,
        activation_history,
        lateral_weight_history,
        output_positions,
        input_positions,
        input_image=input_image,
        input_features=input_features,
        output_dir=output_dir,
        fps=args.fps,
        dpi=args.dpi,
        min_connection=args.min_connection,
        connection_threshold=args.connection_threshold,
        lateral_threshold=args.lateral_threshold,
        layout=args.layout
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
