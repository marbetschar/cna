"""
Enhanced 3D Network Structure Visualization with Input Features

This script creates an enhanced 3D visualization that shows:
1. The network as a 3D grid structure with neurons as points
2. Connections between neurons as lines with color/width representing strength
3. The input image
4. The feature maps from S1 (feature extractor)
5. Animation over training epochs showing evolution

The visualization is designed to give an intuitive understanding of how
the network structure develops in response to the input.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
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
    parser = argparse.ArgumentParser(description='Enhanced 3D Network Structure Visualization')
    parser.add_argument("config", type=str, help="Config file name (e.g., net-fragments)")
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train and visualize (default: 5)')
    parser.add_argument('--output_dir', type=str, default="../tmp/3d_enhanced/",
                        help='Directory to save the visualization outputs')
    parser.add_argument('--video_name', type=str, default="network_evolution.mp4",
                        help='Name of the output video file')
    parser.add_argument('--dpi', type=int, default=200,
                        help='DPI for the video (default: 200)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for the video (default: 5)')
    parser.add_argument('--train_images', type=int, default=2,
                        help='Number of training images (default: 2)')
    parser.add_argument('--test_images', type=int, default=1,
                        help='Number of test images (default: 1)')
    parser.add_argument('--connection_threshold', type=float, default=0.05,
                        help='Minimum weight to show as a strong connection (default: 0.05)')
    parser.add_argument('--min_connection', type=float, default=0.001,
                        help='Minimum weight to show any connection at all (default: 0.001)')
    parser.add_argument('--layout', type=str, default='circle',
                        choices=['circle', 'grid', 'layers'],
                        help='Layout for input neurons (default: circle)')
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
    """
    # Output neurons at z=0 in a grid
    cols_out = int(np.ceil(np.sqrt(n_out)))
    rows_out = int(np.ceil(n_out / cols_out))
    spacing_out = 1.0
    
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
) -> Tuple[List[np.ndarray], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Run training and capture network state at each epoch.
    
    Returns:
        weight_history: List of 2D weight matrices (out, in)
        input_image: Sample input image
        input_features: Sample input features
    """
    feature_extractor.eval()
    lateral_network.eval()
    
    weight_history = []
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
                for t in range(config["lateral_model"]["max_timesteps"]):
                    lateral_network.model.update_ts(t)
                    x_in = torch.cat([x_view_features, z], dim=1)
                    z_float, z = lateral_network(x_in)
                    features_lat.append(z)
                
                features_lat = torch.stack(features_lat, dim=1)
                features_lat_median = torch.median(features_lat, dim=1)[0]
                
                # Train
                x_rearranged = lateral_network.model.s2.rearrange_input(
                    torch.cat([x_view_features, features_lat_median], dim=1))
                lateral_network.model.s2.hebbian_update(x_rearranged, features_lat_median)
        
        # Capture weights
        weights_4d = get_weights_3d(lateral_network)
        weights_2d = extract_all_connections(weights_4d)
        weight_history.append(weights_2d.copy())
        
        print(f"  Weights: min={weights_2d.min():.6f}, max={weights_2d.max():.6f}, mean={weights_2d.mean():.6f}")
        n_strong = np.sum(weights_2d > 0.1)
        print(f"  Strong connections (>0.1): {n_strong}/{weights_2d.size}")
    
    return weight_history, input_image, input_features


def create_enhanced_visualization(
    weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    input_positions: np.ndarray,
    input_image: Optional[torch.Tensor] = None,
    input_features: Optional[torch.Tensor] = None,
    output_dir: Path = None,
    fps: int = 5,
    dpi: int = 200,
    min_connection: float = 0.001,
    connection_threshold: float = 0.05,
    layout: str = 'circle'
):
    """
    Create the enhanced visualization with multiple views.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Creating Enhanced 3D Network Visualization")
    print("="*70)
    
    # 1. Main 3D network structure video with input
    print("\n1. Creating main 3D network structure video...")
    create_3d_network_video(
        weight_history,
        output_positions,
        input_positions,
        input_image=input_image,
        input_features=input_features,
        output_path=output_dir / "network_structure_3d.mp4",
        fps=fps,
        dpi=dpi,
        min_connection=min_connection,
        connection_threshold=connection_threshold
    )
    
    # 2. 2D connection matrix video
    print("\n2. Creating 2D connection matrix video...")
    create_2d_matrix_video(
        weight_history,
        output_dir / "connection_matrix_2d.mp4",
        fps=fps,
        dpi=dpi
    )
    
    # 3. Input image and features
    if input_image is not None:
        print("\n3. Saving input image and features...")
        save_input_visualization(
            input_image,
            input_features,
            output_dir / "input_features.png",
            dpi=dpi
        )
    
    # 4. Connection statistics
    print("\n5. Creating connection statistics...")
    create_connection_stats(
        weight_history,
        output_dir / "connection_stats.png",
        dpi=dpi
    )
    
    print("\n" + "="*70)
    print("All visualizations completed!")
    print(f"Output directory: {output_dir}")
    print("="*70)


def create_3d_network_video(
    weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    input_positions: np.ndarray,
    input_image: Optional[torch.Tensor] = None,
    input_features: Optional[torch.Tensor] = None,
    output_path: Path = None,
    fps: int = 5,
    dpi: int = 200,
    min_connection: float = 0.001,
    connection_threshold: float = 0.05
):
    """
    Create the main 3D network structure video.
    
    The video shows:
    - 3D grid with output neurons (blue) and input neurons (red)
    - Connections as lines with color (plasma colormap) and width representing strength
    - Input image on the side
    - Feature maps from S1
    """
    print(f"  Creating video with {len(output_positions)} output and {len(input_positions)} input neurons")
    print(f"  Frames: {len(weight_history)}")
    
    if output_path is None:
        output_path = Path("network_structure_3d.mp4")
    
    # Get dimensions
    n_out = len(output_positions)
    n_in = len(input_positions)
    n_frames = len(weight_history)
    
    # Find global min/max weights across all frames
    all_weights = np.concatenate([w.flatten() for w in weight_history])
    global_min = all_weights.min()
    global_max = all_weights.max()
    
    # Create figure with 3 subplots:
    # Left: 3D network structure
    # Top-right: Input image
    # Bottom-right: Feature maps or activations
    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_input = fig.add_subplot(gs[0, 1])
    ax_features = fig.add_subplot(gs[1, 1])
    
    # Set up 3D view
    ax_3d.view_init(elev=20, azim=-45)
    ax_3d.set_xlabel('X', fontsize=12, labelpad=10)
    ax_3d.set_ylabel('Y', fontsize=12, labelpad=10)
    ax_3d.set_zlabel('Z', fontsize=12, labelpad=10)
    
    # Set axis limits
    all_positions = np.vstack([output_positions, input_positions])
    x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
    y_min, y_max = all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1
    z_min, z_max = all_positions[:, 2].min() - 1, all_positions[:, 2].max() + 1
    
    ax_3d.set_xlim([x_min, x_max])
    ax_3d.set_ylim([y_min, y_max])
    ax_3d.set_zlim([z_min, z_max])
    
    # Plot output neurons (blue points)
    output_scatter = ax_3d.scatter(
        output_positions[:, 0],
        output_positions[:, 1],
        output_positions[:, 2],
        c='blue',
        s=80,
        alpha=0.9,
        depthshade=True,
        label='Output Neurons'
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
        label='Input Neurons'
    )
    
    # Create all possible connection lines
    lines = []
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
            lines.append(line)
    
    # Show input image
    if input_image is not None:
        img_array = input_image[0, 0].cpu().numpy() if input_image.is_cuda else input_image[0, 0].numpy()
        img_input = ax_input.imshow(img_array, cmap='gray', interpolation='nearest', aspect='auto')
        ax_input.set_title('Input Image', fontsize=12)
        ax_input.axis('off')
    else:
        ax_input.axis('off')
    
    # Show features
    if input_features is not None:
        # Show first 4 feature maps
        feat_array = input_features[0, :, 0].cpu().numpy() if input_features.is_cuda else input_features[0, :, 0].numpy()
        n_feats = min(4, feat_array.shape[0])
        
        # Normalize each feature map
        feat_norm = (feat_array - feat_array.min()) / (feat_array.max() - feat_array.min() + 1e-10)
        
        # Display as a grid
        grid_size = int(np.ceil(np.sqrt(n_feats)))
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
    else:
        ax_features.axis('off')
    
    # Set initial title
    ax_3d.set_title(f'3D Network Structure - Epoch 1\nWeights: [{global_min:.4f}, {global_max:.4f}]', fontsize=14)
    
    # Add colorbar
    dummy_mappable = cm.ScalarMappable(cmap=cm.plasma, norm=Normalize(vmin=0, vmax=1))
    dummy_mappable.set_array([])
    cbar = fig.colorbar(dummy_mappable, ax=ax_3d, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('Connection Strength', fontsize=12)
    
    # Add legend
    ax_3d.legend(loc='upper right')
    
    fig.tight_layout()
    
    def update(frame: int):
        """Update function for animation."""
        weights_2d = weight_history[frame]
        
        # Normalize current weights
        current_min = weights_2d.min()
        current_max = weights_2d.max()
        
        if current_max - current_min > 1e-10:
            weights_norm = (weights_2d - current_min) / (current_max - current_min)
        else:
            weights_norm = np.zeros_like(weights_2d)
        
        # Update all connection lines
        line_idx = 0
        max_linewidth = 2.5
        for i in range(n_out):
            for j in range(n_in):
                weight = weights_2d[i, j]
                
                if line_idx < len(lines):
                    if weight >= min_connection:
                        # Normalize weight for color and width
                        if current_max - current_min > 1e-10:
                            norm_weight = (weight - current_min) / (current_max - current_min)
                        else:
                            norm_weight = 0
                        
                        # Color from plasma colormap
                        color = cm.plasma(norm_weight)
                        
                        # Width based on weight
                        if weight >= connection_threshold:
                            linewidth = 0.3 + norm_weight * max_linewidth
                            alpha = 1.0
                        else:
                            linewidth = 0.1 + norm_weight * 0.3
                            alpha = 0.3
                        
                        lines[line_idx].set_color(color)
                        lines[line_idx].set_linewidth(linewidth)
                        lines[line_idx].set_alpha(alpha)
                        lines[line_idx].set_zorder(2)
                    else:
                        lines[line_idx].set_color('gray')
                        lines[line_idx].set_linewidth(0.05)
                        lines[line_idx].set_alpha(0.05)
                        lines[line_idx].set_zorder(1)
                line_idx += 1
        
        # Update title
        n_strong = np.sum(weights_2d > connection_threshold)
        n_weak = np.sum((weights_2d > min_connection) & (weights_2d <= connection_threshold))
        ax_3d.set_title(
            f'3D Network Structure - Epoch {frame + 1}\n' \
            f'Strong connections: {n_strong}, Weak connections: {n_weak}\n' \
            f'Weights: [{current_min:.4f}, {current_max:.4f}]', 
            fontsize=12
        )
        
        # Update feature display with epoch info
        ax_features.clear()
        ax_features.text(0.5, 0.5, f'Epoch {frame + 1}', 
                        ha='center', va='center', fontsize=20)
        ax_features.axis('off')
        ax_features.set_title(f'Training Progress', fontsize=12)
        
        return lines + [output_scatter, input_scatter, ax_3d]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames, 
                       interval=1000/fps, blit=False, repeat=False)
    
    # Save as video
    writer = FFMpegWriter(fps=fps, bitrate=25000, metadata={'title': '3D Network Structure'})
    ani.save(str(output_path), writer=writer)
    
    plt.close(fig)
    print(f"  Video saved to: {output_path}")
    
    return output_path


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
    
    fig = plt.figure(figsize=(12, 10), dpi=dpi)
    ax = fig.add_subplot(111)
    
    img = ax.imshow(
        matrices_array[0],
        cmap='plasma',
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest',
        aspect='auto'
    )
    
    ax.set_xlabel('Input Neurons', fontsize=14)
    ax.set_ylabel('Output Neurons', fontsize=14)
    ax.set_title(f'Connection Matrix - Epoch 1', fontsize=16)
    
    fig.colorbar(img, ax=ax, shrink=0.7)
    fig.tight_layout()
    
    def update(frame):
        img.set_array(matrices_array[frame])
        current_min = matrices_array[frame].min()
        current_max = matrices_array[frame].max()
        n_strong = np.sum(matrices_array[frame] > 0.1)
        ax.set_title(f'Connection Matrix - Epoch {frame + 1}\nRange: [{current_min:.4f}, {current_max:.4f}], Strong: {n_strong}', 
                    fontsize=14)
        return img, ax
    
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    writer = FFMpegWriter(fps=fps, bitrate=15000)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    
    print(f"  2D matrix video saved to: {output_path}")


def save_input_visualization(
    input_image: torch.Tensor,
    input_features: torch.Tensor,
    output_path: Path,
    dpi: int = 200
):
    """Save input image and feature maps."""
    fig = plt.figure(figsize=(14, 6), dpi=dpi)
    
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


def create_connection_stats(weight_history: List[np.ndarray], output_path: Path, dpi: int = 200):
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
    
    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    
    # Plot 1: Average and max weights
    ax1 = fig.add_subplot(2, 2, 1)
    epochs = range(1, n_frames + 1)
    ax1.plot(epochs, avg_weights, 'b-', linewidth=2, label='Average Weight')
    ax1.plot(epochs, max_weights, 'r-', linewidth=2, label='Max Weight')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weight Value', fontsize=12)
    ax1.set_title('Weight Statistics Over Training', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of connections
    ax2 = fig.add_subplot(2, 2, 2)
    colors = ['g', 'c', 'b', 'm', 'r']
    for idx, t in enumerate(thresholds):
        ax2.plot(epochs, n_connections[t], f'{colors[idx]}-', linewidth=2, label=f'> {t}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Number of Connections Above Threshold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sparsity
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(epochs, sparsity, 'm-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Sparsity (Weights < 0.001)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weight distribution at last epoch
    ax4 = fig.add_subplot(2, 2, 4)
    last_weights = weight_history[-1].flatten()
    ax4.hist(last_weights[last_weights > 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black', range=(0, 1))
    ax4.set_xlabel('Weight Value', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'Weight Distribution (Epoch {n_frames})', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Network Connection Statistics', fontsize=16)
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
    
    # Override dataset sizes
    if args.train_images is not None:
        config['dataset']['train_dataset_params']['num_images'] = args.train_images
    if args.test_images is not None:
        config['dataset']['test_dataset_params']['num_images'] = args.test_images
    
    print("="*70)
    print("Enhanced 3D Network Structure Visualization")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Connection threshold: {args.connection_threshold}")
    print(f"Min connection: {args.min_connection}")
    print(f"Layout: {args.layout}")
    print(f"Train images: {config['dataset']['train_dataset_params']['num_images']}")
    print(f"Test images: {config['dataset']['test_dataset_params']['num_images']}")
    
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
    print(f"  Total possible connections: {n_out * n_in}")
    
    # Create neuron positions
    output_positions, input_positions = create_network_positions(
        n_out, n_in, layout=args.layout
    )
    
    # Run training and capture state
    print("\nRunning training and capturing network state...")
    weight_history, input_image, input_features = run_training_and_capture(
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
        output_positions,
        input_positions,
        input_image=input_image,
        input_features=input_features,
        output_dir=output_dir,
        fps=args.fps,
        dpi=args.dpi,
        min_connection=args.min_connection,
        connection_threshold=args.connection_threshold,
        layout=args.layout
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
