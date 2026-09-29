"""
3D Network Structure Visualization with Input Image

This script creates a 3D visualization of the neural network as a grid structure
where neurons are points in 3D space and connections between them are lines
with thickness/color representing connection strength. The visualization animates
over training epochs to show how the network structure evolves.

Additionally, the input image is displayed alongside the network to show the
relationship between the input and the evolving network structure.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
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
    parser = argparse.ArgumentParser(description='3D Network Structure Visualization')
    parser.add_argument("config", type=str, help="Path to the config file (e.g., net-fragments)")
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of epochs to train and visualize (default: 5)')
    parser.add_argument('--output_dir', type=str, default="../tmp/3d_structure/",
                        help='Directory to save the visualization outputs')
    parser.add_argument('--video_name', type=str, default="network_structure_3d.mp4",
                        help='Name of the output video file')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for the video (default: 150)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for the video (default: 5)')
    parser.add_argument('--train_images', type=int, default=1,
                        help='Number of training images to use (default: 1)')
    parser.add_argument('--test_images', type=int, default=1,
                        help='Number of test images (default: 1)')
    parser.add_argument('--show_input', action='store_true', default=True,
                        help='Show input image in visualization')
    parser.add_argument('--neuron_scale', type=float, default=0.8,
                        help='Scale of neuron markers (default: 0.8)')
    parser.add_argument('--connection_threshold', type=float, default=0.1,
                        help='Minimum weight to show a connection (default: 0.1)')
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


def get_weights_3d(lateral_network: LateralNetwork) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract the 3D weight matrix from the lateral network.
    
    Returns:
        weights_4d: 4D numpy array (out_channels, in_channels, kernel_h, kernel_w)
        info: Dictionary with network dimensions
    """
    s2_layer = lateral_network.model.s2
    weights_3d = s2_layer.get_weights()
    weights_np = weights_3d.detach().cpu().numpy()
    
    info = {
        'out_channels': s2_layer.out_channels,
        'in_channels': s2_layer.in_channels,
        'kernel_size': s2_layer.kernel_size,
        'locality_size': s2_layer.locality_size,
    }
    
    return weights_np, info


def create_3d_grid_positions(n_out: int, n_in: int, kernel_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D positions for neurons in a grid structure.
    
    Args:
        n_out: Number of output neurons
        n_in: Number of input neurons  
        kernel_size: (height, width) of the kernel
    
    Returns:
        output_positions: (n_out, 3) array of 3D positions for output neurons
        input_positions: (n_in, 3) array of 3D positions for input neurons
    """
    # Arrange output neurons in a grid
    # Try to make it as square as possible
    cols_out = int(np.ceil(np.sqrt(n_out)))
    rows_out = int(np.ceil(n_out / cols_out))
    
    # Spacing between neurons
    spacing = 1.0
    
    # Output neuron positions (on one plane)
    output_positions = np.zeros((n_out, 3))
    for i in range(n_out):
        row = i // cols_out
        col = i % cols_out
        output_positions[i, 0] = col * spacing
        output_positions[i, 1] = row * spacing
        output_positions[i, 2] = 0  # All on z=0 plane
    
    # Center the output neurons
    output_positions[:, 0] -= output_positions[:, 0].mean()
    output_positions[:, 1] -= output_positions[:, 1].mean()
    
    # Input neuron positions (on a parallel plane behind)
    cols_in = int(np.ceil(np.sqrt(n_in)))
    rows_in = int(np.ceil(n_in / cols_in))
    
    input_positions = np.zeros((n_in, 3))
    for i in range(n_in):
        row = i // cols_in
        col = i % cols_in
        input_positions[i, 0] = col * spacing
        input_positions[i, 1] = row * spacing
        input_positions[i, 2] = -3.0  # Behind output neurons
    
    # Center the input neurons
    input_positions[:, 0] -= input_positions[:, 0].mean()
    input_positions[:, 1] -= input_positions[:, 1].mean()
    
    return output_positions, input_positions


def create_3d_grid_flat(n_out: int, n_in: int, kernel_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D positions where neurons are arranged in a more interesting 3D structure.
    
    Output neurons are on one plane, input neurons are distributed in 3D space
    based on their channel and spatial position.
    """
    kh, kw = kernel_size
    
    # Output neuron positions - arranged in a 2D grid on the front plane (z=0)
    cols_out = int(np.ceil(np.sqrt(n_out)))
    rows_out = int(np.ceil(n_out / cols_out))
    spacing_out = 1.2
    
    output_positions = np.zeros((n_out, 3))
    for i in range(n_out):
        row = i // cols_out
        col = i % cols_out
        output_positions[i, 0] = col * spacing_out
        output_positions[i, 1] = row * spacing_out
        output_positions[i, 2] = 0
    
    # Center
    output_positions[:, 0] -= output_positions[:, 0].mean()
    output_positions[:, 1] -= output_positions[:, 1].mean()
    
    # Input neuron positions - arranged in 3D space behind
    # We'll arrange them in a cylindrical pattern around the output neurons
    spacing_in = 0.6
    radius = 3.0  # Distance behind output neurons
    
    input_positions = np.zeros((n_in, 3))
    for i in range(n_in):
        # Distribute in a circle in the x-y plane at z = -radius
        angle = 2 * np.pi * i / n_in
        input_positions[i, 0] = radius * np.cos(angle)
        input_positions[i, 1] = radius * np.sin(angle)
        input_positions[i, 2] = -radius
    
    # Add some vertical variation based on channel
    for i in range(n_in):
        ch = i % 4  # 4 input channels from feature extractor
        input_positions[i, 2] -= ch * 0.5
    
    return output_positions, input_positions


def extract_connections(weights_4d: np.ndarray, threshold: float = 0.1, return_all: bool = False) -> List[Tuple[int, int, float]]:
    """
    Extract connections from the weight matrix.
    
    Args:
        weights_4d: (out_channels, in_channels, kernel_h, kernel_w)
        threshold: Minimum weight to consider as a connection (if return_all=False)
        return_all: If True, return all connections regardless of weight
    
    Returns:
        List of (output_idx, input_idx, weight) tuples for connections
    """
    connections = []
    out_channels, in_channels, kh, kw = weights_4d.shape
    
    # Flatten spatial dimensions
    for out_idx in range(out_channels):
        for in_idx in range(in_channels):
            # Get all weights between this output and input neuron
            weights_flat = weights_4d[out_idx, in_idx].flatten()
            
            # Use the mean weight across spatial positions
            mean_weight = weights_flat.mean()
            
            if return_all or mean_weight >= threshold:
                connections.append((out_idx, in_idx, float(mean_weight)))
    
    return connections


def normalize_weights(connections: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """Normalize connection weights to [0, 1] range."""
    if not connections:
        return connections
    
    weights = [c[2] for c in connections]
    min_w = min(weights)
    max_w = max(weights)
    
    if max_w - min_w < 1e-10:
        normalized = [(c[0], c[1], 0.5) for c in connections]
    else:
        normalized = [(c[0], c[1], (c[2] - min_w) / (max_w - min_w)) for c in connections]
    
    return normalized


class NetworkStructureVisualizer:
    """
    Main class for creating 3D network structure visualizations.
    """
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Override dataset sizes
        if args.train_images is not None:
            config['dataset']['train_dataset_params']['num_images'] = args.train_images
        if args.test_images is not None:
            config['dataset']['test_dataset_params']['num_images'] = args.test_images
        
        # Setup
        self.fabric = setup_fabric(config)
        self.train_loader, self.test_loader = setup_dataloader(config, self.fabric)
        self.feature_extractor = setup_feature_extractor(config, self.fabric)
        self.lateral_network = setup_lateral_network(config, self.fabric)
        
        # Network info
        self.s2_layer = self.lateral_network.model.s2
        self.out_channels = self.s2_layer.out_channels
        self.in_channels = self.s2_layer.in_channels
        self.kernel_size = self.s2_layer.kernel_size
        
        # Store data for visualization
        self.weight_history = []
        self.input_image = None
        self.features = None
        
        print(f"Network Architecture:")
        print(f"  S2 Lateral Layer: {self.out_channels} output channels, {self.in_channels} input channels")
        print(f"  Kernel size: {self.kernel_size}")
        print(f"  Total connections: {self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]}")
        
        # Create 3D positions for neurons
        self.output_positions, self.input_positions = create_3d_grid_flat(
            self.out_channels, self.in_channels, self.kernel_size
        )


def get_single_batch(feature_extractor, data_loader):
    """Get a single batch from the dataloader."""
    for batch in data_loader:
        return batch[0]
    return None


def run_training_and_capture(
    config: Dict[str, Any],
    feature_extractor,
    lateral_network: LateralNetwork,
    train_loader,
    epochs: int,
    capture_input: bool = True
) -> Tuple[List[np.ndarray], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Run training and capture weight matrices and input features at each epoch.
    
    Returns:
        weight_history: List of weight arrays (4D) at each epoch
        input_image: The input image (if captured)
        features: The extracted features (if captured)
    """
    feature_extractor.eval()
    lateral_network.eval()
    
    weight_history = []
    input_image = None
    features = None
    
    # Get a sample input for visualization
    if capture_input:
        sample_batch = get_single_batch(feature_extractor, train_loader)
        if sample_batch is not None:
            with torch.no_grad():
                input_image = sample_batch.clone()
                features = feature_extractor(sample_batch)
    
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
        
        # Capture weights at end of epoch
        weights_4d, _ = get_weights_3d(lateral_network)
        weight_history.append(weights_4d.copy())
        
        print(f"  Weights: min={weights_4d.min():.4f}, max={weights_4d.max():.4f}, mean={weights_4d.mean():.4f}")
    
    return weight_history, input_image, features


def create_network_structure_video(
    weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    input_positions: np.ndarray,
    input_image: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    output_path: Path = None,
    fps: int = 5,
    dpi: int = 150,
    connection_threshold: float = 0.1,
    show_input: bool = True,
    neuron_scale: float = 0.8
):
    """
    Create a 3D animated video of the network structure.
    
    The video shows:
    - Output neurons as points on one plane
    - Input neurons as points on another plane
    - Connections as lines between neurons with color/width representing strength
    - Input image displayed alongside (optional)
    """
    print(f"\nCreating network structure video...")
    print(f"  Frames: {len(weight_history)}")
    print(f"  Output neurons: {len(output_positions)}")
    print(f"  Input neurons: {len(input_positions)}")
    
    if output_path is None:
        output_path = Path("network_structure.mp4")
    
    # Setup figure
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    
    # Create subplots: 3D network on left, input image on right
    if show_input and input_image is not None:
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_img = fig.add_subplot(122)
    else:
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_img = None
    
    # Set viewing angle for 3D plot
    ax_3d.view_init(elev=20, azim=-45)
    ax_3d.set_xlabel('X', fontsize=12)
    ax_3d.set_ylabel('Y', fontsize=12)
    ax_3d.set_zlabel('Z', fontsize=12)
    
    # Set axis limits
    all_positions = np.vstack([output_positions, input_positions])
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min() - 2, all_positions[:, 2].max() + 2
    
    ax_3d.set_xlim([x_min - 1, x_max + 1])
    ax_3d.set_ylim([y_min - 1, y_max + 1])
    ax_3d.set_zlim([z_min, z_max])
    
    # Create initial plots
    n_out = len(output_positions)
    n_in = len(input_positions)
    
    # Plot output neurons (as larger points)
    output_scatter = ax_3d.scatter(
        output_positions[:, 0],
        output_positions[:, 1],
        output_positions[:, 2],
        c='blue',
        s=50 * neuron_scale,
        alpha=0.8,
        depthshade=True
    )
    
    # Plot input neurons (as smaller points)
    input_scatter = ax_3d.scatter(
        input_positions[:, 0],
        input_positions[:, 1],
        input_positions[:, 2],
        c='red',
        s=30 * neuron_scale,
        alpha=0.8,
        depthshade=True
    )
    
    # Initial connections (empty)
    lines = []
    for i in range(n_out):
        for j in range(n_in):
            line = ax_3d.plot(
                [output_positions[i, 0], input_positions[j, 0]],
                [output_positions[i, 1], input_positions[j, 1]],
                [output_positions[i, 2], input_positions[j, 2]],
                'k-',
                linewidth=0.1,
                alpha=0.0
            )[0]
            lines.append(line)
    
    # Show input image
    if show_input and input_image is not None and ax_img is not None:
        img_array = input_image[0, 0].cpu().numpy() if input_image.is_cuda else input_image[0, 0].numpy()
        img_display = ax_img.imshow(img_array, cmap='gray', interpolation='nearest')
        ax_img.set_title('Input Image', fontsize=14)
        ax_img.axis('off')
    
    # Set title
    ax_3d.set_title(f'3D Network Structure - Epoch 0\nConnections: 0, Weights: [0, 0]', fontsize=14)
    
    # Colorbar for connection strengths
    # We'll create a dummy mappable for the colorbar
    dummy_mappable = cm.ScalarMappable(cmap=cm.plasma, norm=Normalize(vmin=0, vmax=1))
    dummy_mappable.set_array([])
    cbar = fig.colorbar(dummy_mappable, ax=ax_3d, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Connection Strength', fontsize=12)
    
    fig.tight_layout()
    
    def update(frame: int):
        """Update function for animation."""
        # Get weights at this epoch
        weights_4d = weight_history[frame]
        
        # Extract connections above threshold
        connections = extract_connections(weights_4d, threshold=connection_threshold)
        connections = normalize_weights(connections)
        
        # Update all connection lines
        line_idx = 0
        max_linewidth = 3.0
        for i in range(n_out):
            for j in range(n_in):
                # Find if there's a connection from i to j
                conn_weight = 0
                for (out_idx, in_idx, weight) in connections:
                    if out_idx == i and in_idx == j:
                        conn_weight = weight
                        break
                
                if line_idx < len(lines):
                    if conn_weight > 0:
                        # Map weight to color (plasma colormap: purple->yellow)
                        color = cm.plasma(conn_weight)
                        linewidth = 0.5 + conn_weight * max_linewidth
                        lines[line_idx].set_color(color)
                        lines[line_idx].set_linewidth(linewidth)
                        lines[line_idx].set_alpha(1.0)
                    else:
                        lines[line_idx].set_color('gray')
                        lines[line_idx].set_linewidth(0.1)
                        lines[line_idx].set_alpha(0.1)
                line_idx += 1
        
        # Update title
        num_connections = len(connections)
        if connections:
            min_w = min(c[2] for c in connections)
            max_w = max(c[2] for c in connections)
        else:
            min_w, max_w = 0, 0
        
        ax_3d.set_title(f'3D Network Structure - Epoch {frame + 1}\nConnections: {num_connections}, Weights: [{min_w:.3f}, {max_w:.3f}]', 
                       fontsize=14)
        
        return lines + [output_scatter, input_scatter]
    
    # Create animation
    n_frames = len(weight_history)
    ani = FuncAnimation(fig, update, frames=n_frames, 
                       interval=1000/fps, blit=False, repeat=False)
    
    # Save as video
    writer = FFMpegWriter(fps=fps, bitrate=20000, metadata={'title': '3D Network Structure'})
    ani.save(str(output_path), writer=writer)
    
    plt.close(fig)
    print(f"Video saved to: {output_path}")
    
    return output_path


def create_combined_visualization(
    weight_history: List[np.ndarray],
    output_positions: np.ndarray,
    input_positions: np.ndarray,
    input_image: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    output_dir: Path = None,
    fps: int = 5,
    dpi: int = 150,
    connection_threshold: float = 0.1,
    neuron_scale: float = 0.8
):
    """
    Create multiple visualizations: network structure video + weight evolution plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating Network Structure Visualization")
    print("="*60)
    
    # 1. Create the main 3D network structure video
    video_path = output_dir / "network_structure_3d.mp4"
    create_network_structure_video(
        weight_history,
        output_positions,
        input_positions,
        input_image=input_image,
        features=features,
        output_path=video_path,
        fps=fps,
        dpi=dpi,
        connection_threshold=connection_threshold,
        neuron_scale=neuron_scale
    )
    
    # 2. Create a 2D connection matrix video
    print("\nCreating 2D connection matrix video...")
    create_2d_matrix_video(
        weight_history,
        output_dir / "connection_matrix_2d.mp4",
        fps=fps,
        dpi=dpi
    )
    
    # 3. Save the input image
    if input_image is not None:
        print("\nSaving input image...")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
        img_array = input_image[0, 0].cpu().numpy() if input_image.is_cuda else input_image[0, 0].numpy()
        ax.imshow(img_array, cmap='gray', interpolation='nearest')
        ax.set_title('Input Image', fontsize=14)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(output_dir / "input_image.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    # 4. Create connection statistics plot
    print("\nCreating connection statistics...")
    create_connection_stats(
        weight_history,
        output_dir / "connection_stats.png",
        dpi=dpi
    )
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print(f"Output directory: {output_dir}")
    print("="*60)


def create_2d_matrix_video(
    weight_history: List[np.ndarray],
    output_path: Path,
    fps: int = 5,
    dpi: int = 150
):
    """Create a 2D video showing the weight matrix evolution."""
    # Flatten weights to 2D
    matrices_2d = []
    for weights_4d in weight_history:
        out, in_ch, kh, kw = weights_4d.shape
        # Average across spatial dimensions
        matrix_2d = weights_4d.mean(axis=(2, 3))
        matrices_2d.append(matrix_2d)
    
    matrices_array = np.stack(matrices_2d, axis=0)
    n_frames, n_out, n_in = matrices_array.shape
    
    global_min = matrices_array.min()
    global_max = matrices_array.max()
    
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Initial image
    img = ax.imshow(
        matrices_array[0],
        cmap='plasma',
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest',
        aspect='auto'
    )
    
    ax.set_xlabel('Input Neurons', fontsize=12)
    ax.set_ylabel('Output Neurons', fontsize=12)
    ax.set_title(f'Connection Matrix - Epoch 1\nRange: [{global_min:.4f}, {global_max:.4f}]', fontsize=14)
    
    # Colorbar
    fig.colorbar(img, ax=ax, shrink=0.7)
    
    fig.tight_layout()
    
    def update(frame):
        img.set_array(matrices_array[frame])
        current_min = matrices_array[frame].min()
        current_max = matrices_array[frame].max()
        ax.set_title(f'Connection Matrix - Epoch {frame + 1}\nRange: [{current_min:.4f}, {current_max:.4f}]', fontsize=12)
        return img, ax
    
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    writer = FFMpegWriter(fps=fps, bitrate=10000)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    
    print(f"  2D matrix video saved to: {output_path}")


def create_connection_stats(weight_history: List[np.ndarray], output_path: Path, dpi: int = 150):
    """Create a plot showing connection statistics over training."""
    # Compute statistics
    n_frames = len(weight_history)
    
    # Average weight
    avg_weights = [w.mean() for w in weight_history]
    
    # Max weight
    max_weights = [w.max() for w in weight_history]
    
    # Number of connections above various thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    n_connections = {}
    for t in thresholds:
        n_connections[t] = [np.sum(w > t) for w in weight_history]
    
    # Sparsity
    sparsity = [np.mean(w < 0.01) * 100 for w in weight_history]
    
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    
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
    
    # Plot 2: Number of connections at different thresholds
    ax2 = fig.add_subplot(2, 2, 2)
    for t, color, label in [(0.1, 'g', '> 0.1'), (0.3, 'b', '> 0.3'), (0.5, 'm', '> 0.5'), (0.7, 'r', '> 0.7')]:
        ax2.plot(epochs, n_connections[t], f'{color}-', linewidth=2, label=f'Connections {label}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Number of Strong Connections', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sparsity
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(epochs, sparsity, 'm-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Sparsity (Weights < 0.01)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weight distribution at last epoch
    ax4 = fig.add_subplot(2, 2, 4)
    last_weights = weight_history[-1].flatten()
    ax4.hist(last_weights, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
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
    
    print("="*60)
    print("3D Network Structure Visualization")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Show input: {args.show_input}")
    print(f"Connection threshold: {args.connection_threshold}")
    
    # Create visualizer
    visualizer = NetworkStructureVisualizer(config, args)
    
    # Run training and capture weights
    print("\nRunning training and capturing network state...")
    weight_history, input_image, features = run_training_and_capture(
        config,
        visualizer.feature_extractor,
        visualizer.lateral_network,
        visualizer.train_loader,
        epochs=args.epochs,
        capture_input=args.show_input
    )
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    create_combined_visualization(
        weight_history,
        visualizer.output_positions,
        visualizer.input_positions,
        input_image=input_image,
        features=features,
        output_dir=output_dir,
        fps=args.fps,
        dpi=args.dpi,
        connection_threshold=args.connection_threshold,
        neuron_scale=args.neuron_scale
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
