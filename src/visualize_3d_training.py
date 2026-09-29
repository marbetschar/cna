"""
3D Visualization of Neural Network Training Process

This script creates a 3D animated heatmap showing the training process
within the neural network, visualizing all neurons and their connection
strengths as a 3D heatmap over training epochs.

The visualization shows:
- X-axis: Input neurons (presynaptic)
- Y-axis: Output neurons (postsynaptic) 
- Z-axis: Training epochs
- Color: Connection strength (weight value)
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video generation
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
from models.s2_fragments import LateralNetwork, LateralLayer
from utils.config import get_config
from utils.custom_print import print_start, print_info
from lightning import Fabric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Visualization of Network Training')
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train and visualize (default: 10)')
    parser.add_argument('--output_dir', type=str, default="../tmp/3d_viz/",
                        help='Directory to save the visualization outputs')
    parser.add_argument('--video_name', type=str, default="training_3d_heatmap.mp4",
                        help='Name of the output video file')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for the video (default: 150)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for the video (default: 5)')
    parser.add_argument('--show_2d_slices', action='store_true',
                        help='Also generate 2D slice images at each epoch')
    parser.add_argument('--neuron_subsample', type=int, default=None,
                        help='Subsample neurons for visualization (None for all)')
    parser.add_argument('--train_images', type=int, default=None,
                        help='Number of training images (default: from config)')
    parser.add_argument('--test_images', type=int, default=None,
                        help='Number of test images (default: from config)')
    return parser.parse_args()


def setup_fabric(config: Dict[str, Any]) -> Fabric:
    """Setup the fabric instance."""
    fabric = Fabric(accelerator="auto", devices=1, loggers=[], callbacks=[])
    fabric.launch()
    fabric.seed_everything(1)
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


def extract_weights_3d(lateral_network: LateralNetwork) -> np.ndarray:
    """
    Extract the 3D weight matrix from the lateral network.
    
    Returns a 3D numpy array of shape (output_neurons, input_neurons, kernel_size_x, kernel_size_y)
    or flattened to (output_neurons, input_neurons * kernel_size_x * kernel_size_y)
    """
    # Get the lateral layer
    s2_layer = lateral_network.model.s2
    
    # Get the weights
    weights_3d = s2_layer.get_weights()  # Shape: (out_channels, in_channels, kernel_h, kernel_w)
    
    # Convert to numpy
    weights_np = weights_3d.detach().cpu().numpy()
    
    return weights_np


def extract_weights_2d(lateral_network: LateralNetwork) -> np.ndarray:
    """
    Extract weights as a 2D matrix by flattening spatial dimensions.
    
    Returns shape: (output_neurons, input_neurons * kernel_size_h * kernel_size_w)
    """
    weights_3d = extract_weights_3d(lateral_network)
    
    # Flatten spatial dimensions: (out, in, h, w) -> (out, in * h * w)
    out_channels = weights_3d.shape[0]
    in_channels = weights_3d.shape[1]
    kernel_size = weights_3d.shape[2] * weights_3d.shape[3]
    
    weights_2d = weights_3d.reshape(out_channels, in_channels * kernel_size)
    
    return weights_2d


class NetworkTrainingVisualizer:
    """
    Main class for creating 3D visualizations of network training.
    """
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Override dataset sizes if specified
        if args.train_images is not None:
            config['dataset']['train_dataset_params']['num_images'] = args.train_images
        if args.test_images is not None:
            config['dataset']['test_dataset_params']['num_images'] = args.test_images
        
        # Setup
        self.fabric = setup_fabric(config)
        self.train_loader, self.test_loader = setup_dataloader(config, self.fabric)
        self.feature_extractor = setup_feature_extractor(config, self.fabric)
        self.lateral_network = setup_lateral_network(config, self.fabric)
        
        # Initialize weight history
        self.weight_history = []
        
        # Network info
        self.s2_layer = self.lateral_network.model.s2
        self.out_channels = self.s2_layer.out_channels
        self.in_channels = self.s2_layer.in_channels
        self.kernel_size = self.s2_layer.kernel_size
        
        print_info(f"Network Architecture:", title="Network Info")
        print_info(f"  S2 Lateral Layer: {self.out_channels} output channels, {self.in_channels} input channels")
        print_info(f"  Kernel size: {self.kernel_size}")
        print_info(f"  Total input connections per output neuron: {self.in_channels * self.kernel_size[0] * self.kernel_size[1]}")
        
    def run_training_epoch(self, epoch: int):
        """Run a single training epoch and store weights."""
        self.feature_extractor.eval()
        self.lateral_network.eval()
        
        # Run training
        for i, batch in enumerate(tqdm(self.train_loader, 
                                        total=len(self.train_loader),
                                        colour="GREEN",
                                        desc=f"Epoch {epoch+1}/{self.args.epochs}")):
            with torch.no_grad():
                features = self.feature_extractor(batch[0])
            
            z = None
            for view_idx in range(features.shape[1]):
                x_view_features = features[:, view_idx, ...]
                
                if z is None:
                    z = torch.zeros((x_view_features.shape[0], self.lateral_network.model.out_channels, 
                                     x_view_features.shape[2], x_view_features.shape[3]), 
                                    device=batch[0].device)
                
                features_lat = []
                for t in range(self.config["lateral_model"]["max_timesteps"]):
                    self.lateral_network.model.update_ts(t)
                    x_in = torch.cat([x_view_features, z], dim=1)
                    z_float, z = self.lateral_network(x_in)
                    features_lat.append(z)
                
                features_lat = torch.stack(features_lat, dim=1)
                features_lat_median = torch.median(features_lat, dim=1)[0]
                
                # Train
                x_rearranged = self.lateral_network.model.s2.rearrange_input(
                    torch.cat([x_view_features, features_lat_median], dim=1))
                self.lateral_network.model.s2.hebbian_update(x_rearranged, features_lat_median)
        
        # Store weights at the end of epoch
        weights_2d = extract_weights_2d(self.lateral_network)
        self.weight_history.append(weights_2d.copy())
        
        return weights_2d


def create_3d_heatmap_video(weight_history: List[np.ndarray], 
                           output_path: Path,
                           fps: int = 5,
                           dpi: int = 150,
                           neuron_subsample: Optional[int] = None):
    """
    Create a 3D animated heatmap video of weight evolution during training.
    
    Args:
        weight_history: List of 2D weight matrices (epoch, output_neurons, input_connections)
        output_path: Path to save the video
        fps: Frames per second
        dpi: Dots per inch
        neuron_subsample: If not None, subsample neurons for visualization
    """
    print_start("Creating 3D Heatmap Video...", title="Video Generation")
    
    # Stack weights: (epoch, out_neurons, in_connections)
    weights_array = np.stack(weight_history, axis=0)
    n_epochs, n_out, n_in = weights_array.shape
    
    print_info(f"Weight array shape: {weights_array.shape}")
    print_info(f"Creating {n_epochs} frames...")
    
    # Apply subsampling if requested
    if neuron_subsample is not None and n_out > neuron_subsample:
        # Select a subset of output neurons
        indices = np.linspace(0, n_out - 1, neuron_subsample, dtype=int)
        weights_array = weights_array[:, indices, :]
        n_out = len(indices)
        print_info(f"Subsampled to {n_out} output neurons")
    
    # Find global min/max for consistent color scaling
    global_min = weights_array.min()
    global_max = weights_array.max()
    print_info(f"Weight range: [{global_min:.4f}, {global_max:.4f}]")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    
    # Create 3D axes
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for the weight matrix
    x = np.arange(n_in)  # Input connections
    y = np.arange(n_out)  # Output neurons
    X, Y = np.meshgrid(x, y)
    
    # Initial surface
    Z = np.zeros_like(X)  # Will use color for weights
    surf = ax.plot_surface(X, Y, Z, 
                          cmap=cm.viridis,
                          vmin=global_min,
                          vmax=global_max,
                          rstride=1,
                          cstride=1)
    
    # Set labels
    ax.set_xlabel('Input Connections', fontsize=14, labelpad=10)
    ax.set_ylabel('Output Neurons', fontsize=14, labelpad=10)
    ax.set_zlabel('Training Epoch', fontsize=14, labelpad=10)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add colorbar
    mappable = cm.ScalarMappable(norm=Normalize(vmin=global_min, vmax=global_max), cmap=cm.viridis)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=20, pad=0.1)
    cbar.set_label('Connection Strength', fontsize=14)
    
    # Set title
    ax.set_title(f'3D Heatmap: Neural Network Connection Strengths Over Training\n' 
                 f'Epoch 0, Weights: [{global_min:.4f}, {global_max:.4f}]', fontsize=14)
    
    # Adjust layout
    fig.tight_layout()
    
    def update(frame: int):
        """Update function for animation."""
        # Get weights at this epoch
        weights_2d = weight_history[frame]
        
        # Apply same subsampling to current frame
        if neuron_subsample is not None and n_out > neuron_subsample:
            indices = np.linspace(0, weights_2d.shape[0] - 1, neuron_subsample, dtype=int)
            weights_2d = weights_2d[indices, :]
        
        # Update surface data
        Z = np.full_like(X, frame)  # Z is the epoch number
        surf._offsets3d = (X, Y, Z)
        surf._facecolors = cm.viridis((weights_2d - global_min) / (global_max - global_min + 1e-10))
        surf.changed()
        
        # Update title
        current_min = weights_2d.min()
        current_max = weights_2d.max()
        ax.set_title(f'3D Heatmap: Neural Network Connection Strengths Over Training\n' 
                     f'Epoch {frame}, Weights: [{current_min:.4f}, {current_max:.4f}]', fontsize=14)
        
        return surf, ax
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_epochs, 
                       interval=1000/fps, blit=False, repeat=False)
    
    # Save as video
    writer = FFMpegWriter(fps=fps, bitrate=20000, metadata={'title': '3D Network Training'})
    ani.save(str(output_path), writer=writer)
    
    plt.close(fig)
    print_info(f"Video saved to: {output_path}", title="Complete")
    
    return output_path


def create_2d_slice_images(weight_history: List[np.ndarray], 
                          output_dir: Path,
                          dpi: int = 150):
    """
    Create 2D slice images at each epoch showing the weight matrix.
    """
    print_start("Creating 2D Slice Images...", title="2D Visualization")
    
    output_dir = output_dir / "2d_slices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_epochs = len(weight_history)
    weights_array = np.stack(weight_history, axis=0)
    
    n_out, n_in = weights_array.shape[1], weights_array.shape[2]
    
    # Find global min/max
    global_min = weights_array.min()
    global_max = weights_array.max()
    
    for epoch in range(n_epochs):
        weights_2d = weight_history[epoch]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=dpi)
        
        im = ax.imshow(weights_2d, cmap='viridis', 
                      vmin=global_min, vmax=global_max,
                      aspect='auto', interpolation='nearest')
        
        ax.set_xlabel('Input Connections', fontsize=14)
        ax.set_ylabel('Output Neurons', fontsize=14)
        ax.set_title(f'Epoch {epoch}: Connection Strength Matrix\n' 
                     f'Range: [{weights_2d.min():.4f}, {weights_2d.max():.4f}]',
                     fontsize=14)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Connection Strength', fontsize=12)
        
        fig.tight_layout()
        
        # Save
        output_path = output_dir / f"weights_epoch_{epoch:03d}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    print_info(f"Saved {n_epochs} 2D slice images to: {output_dir}", title="Complete")


def create_weight_distribution_plot(weight_history: List[np.ndarray], 
                                   output_dir: Path,
                                   dpi: int = 150):
    """
    Create a plot showing the distribution of weights over training.
    """
    print_start("Creating Weight Distribution Plot...", title="Distribution Analysis")
    
    weights_array = np.stack(weight_history, axis=0)
    n_epochs = weights_array.shape[0]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    
    # Plot 1: Mean weight over epochs
    ax1 = fig.add_subplot(2, 2, 1)
    mean_weights = weights_array.mean(axis=(1, 2))
    ax1.plot(range(n_epochs), mean_weights, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Weight', fontsize=12)
    ax1.set_title('Mean Connection Strength Over Training', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max weight over epochs
    ax2 = fig.add_subplot(2, 2, 2)
    max_weights = weights_array.max(axis=(1, 2))
    ax2.plot(range(n_epochs), max_weights, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Max Weight', fontsize=12)
    ax2.set_title('Max Connection Strength Over Training', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of strong connections (> 0.5)
    ax3 = fig.add_subplot(2, 2, 3)
    strong_connections = np.sum(weights_array > 0.5, axis=(1, 2))
    ax3.plot(range(n_epochs), strong_connections, 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Number of Strong Connections (> 0.5) Over Training', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sparsity (percentage of near-zero weights)
    ax4 = fig.add_subplot(2, 2, 4)
    sparsity = (weights_array < 0.01).mean(axis=(1, 2)) * 100
    ax4.plot(range(n_epochs), sparsity, 'm-', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_title('Sparsity (Weights < 0.01) Over Training', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    output_path = output_dir / "weight_distribution.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print_info(f"Weight distribution plot saved to: {output_path}", title="Complete")


def create_neuron_activation_heatmap(lateral_network: LateralNetwork,
                                    feature_extractor,
                                    test_loader,
                                    output_dir: Path,
                                    dpi: int = 150):
    """
    Create a heatmap showing neuron activations for a sample input.
    """
    print_start("Creating Neuron Activation Heatmap...", title="Neuron Activation")
    
    lateral_network.eval()
    feature_extractor.eval()
    
    # Get a sample batch
    for batch in test_loader:
        sample_images = batch[0]
        break
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(sample_images)
    
    # Run through lateral network
    z = None
    activations = []
    
    for view_idx in range(features.shape[1]):
        x_view_features = features[:, view_idx, ...]
        
        if z is None:
            z = torch.zeros((x_view_features.shape[0], lateral_network.model.out_channels, 
                             x_view_features.shape[2], x_view_features.shape[3]), 
                            device=sample_images.device)
        
        for t in range(lateral_network.conf["lateral_model"]["max_timesteps"]):
            lateral_network.model.update_ts(t)
            x_in = torch.cat([x_view_features, z], dim=1)
            z_float, z = lateral_network(x_in)
            activations.append(z.detach().cpu().numpy())
    
    # Stack activations
    activations = np.concatenate(activations, axis=1)  # (batch, timesteps * views, channels, h, w)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=dpi)
    
    # Plot activations for first few channels
    n_channels = min(4, activations.shape[2])
    for i, ax in enumerate(axes.flat):
        if i < n_channels:
            # Average across time and batch
            act_map = activations[:, :, i].mean(axis=(0, 1))
            im = ax.imshow(act_map, cmap='hot', interpolation='nearest')
            ax.set_title(f'Neuron {i} Activation Map', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.axis('off')
    
    fig.suptitle('Neuron Activation Maps (Average across time and samples)', fontsize=16)
    fig.tight_layout()
    
    output_path = output_dir / "neuron_activations.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print_info(f"Neuron activation heatmap saved to: {output_path}", title="Complete")


def main():
    """Main function to run the 3D visualization."""
    args = parse_args()
    
    # Update config with command line args
    config = get_config(args.config)
    config['run']['n_epochs'] = args.epochs
    
    print_start("3D Network Training Visualization", 
                title="="*50)
    print_info(f"Configuration: {args.config}")
    print_info(f"Output directory: {args.output_dir}")
    print_info(f"Number of epochs: {args.epochs}")
    print_info(f"Video name: {args.video_name}")
    
    # Create visualizer
    visualizer = NetworkTrainingVisualizer(config, args)
    
    print_start("Running training and collecting weights...", title="Training")
    
    # Run training and collect weights
    for epoch in range(args.epochs):
        weights_2d = visualizer.run_training_epoch(epoch)
        print_info(f"Epoch {epoch + 1}: Weight shape = {weights_2d.shape}, "
                   f"min = {weights_2d.min():.4f}, max = {weights_2d.max():.4f}")
    
    # Create output paths
    video_path = visualizer.output_dir / args.video_name
    
    print_start("Generating visualizations...", title="Visualization")
    
    # Create 3D video
    create_3d_heatmap_video(
        visualizer.weight_history,
        video_path,
        fps=args.fps,
        dpi=args.dpi,
        neuron_subsample=args.neuron_subsample
    )
    
    # Create 2D slices if requested
    if args.show_2d_slices:
        create_2d_slice_images(visualizer.weight_history, visualizer.output_dir, args.dpi)
    
    # Create distribution plot
    create_weight_distribution_plot(visualizer.weight_history, visualizer.output_dir, args.dpi)
    
    # Create neuron activation heatmap
    create_neuron_activation_heatmap(
        visualizer.lateral_network,
        visualizer.feature_extractor,
        visualizer.test_loader,
        visualizer.output_dir,
        args.dpi
    )
    
    print_start("All visualizations completed!", title="="*50)
    print_info(f"Video saved to: {video_path}")
    print_info(f"All outputs in: {visualizer.output_dir}")


if __name__ == '__main__':
    main()
