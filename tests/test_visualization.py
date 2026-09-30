"""
Simple test script to verify the 3D visualization works.
Runs a minimal training and generates a small test visualization.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize

from data.loader import loaders_from_config
from models.s1 import FixedFilterFeatureExtractor
from models.s2_fragments import LateralNetwork
from utils.config import get_config
from lightning import Fabric
from tqdm import tqdm


def setup_simple_training(config_name="net-fragments", epochs=3):
    """Setup a simple training run for testing."""
    print("Setting up simple training test...")
    
    # Load config
    config = get_config(config_name)
    config['run']['n_epochs'] = epochs
    config['run']['batch_size'] = 1
    config['dataset']['train_dataset_params']['num_images'] = 10
    config['dataset']['test_dataset_params']['num_images'] = 5
    
    # Setup fabric
    fabric = Fabric(accelerator="cpu", devices=1, loggers=[], callbacks=[])
    fabric.launch()
    fabric.seed_everything(42)
    
    # Setup dataloaders
    train_loader, test_loader, _ = loaders_from_config(config)
    train_loader = fabric.setup_dataloaders(train_loader)
    
    # Setup models
    feature_extractor = FixedFilterFeatureExtractor(config, fabric)
    feature_extractor = fabric.setup(feature_extractor)
    
    lateral_network = fabric.setup(LateralNetwork(config, fabric))
    
    return config, fabric, train_loader, test_loader, feature_extractor, lateral_network


def get_weights_2d(lateral_network):
    """Get weights as 2D array."""
    s2_layer = lateral_network.model.s2
    weights_3d = s2_layer.get_weights()
    weights_np = weights_3d.detach().cpu().numpy()
    
    # Flatten spatial dimensions
    weights_2d = weights_3d.reshape(weights_3d.shape[0], -1)
    return weights_2d


def run_single_epoch(config, feature_extractor, lateral_network, train_loader, epoch_idx):
    """Run a single training epoch."""
    feature_extractor.eval()
    lateral_network.eval()
    
    for i, batch in enumerate(train_loader):
        with torch.no_grad():
            features = feature_extractor(batch[0])
        
        z = None
        for view_idx in range(features.shape[1]):
            x_view_features = features[:, view_idx, ...]
            
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
    
    # Return weights
    return get_weights_2d(lateral_network)


def create_test_visualization(weight_history, output_path="../tmp/test_3d_small.mp4"):
    """Create a simple 3D test visualization."""
    print("Creating test visualization...")
    
    weights_array = np.stack(weight_history, axis=0)
    n_epochs, n_out, n_in = weights_array.shape
    
    print(f"Weight array shape: {weights_array.shape}")
    
    # Use a subsample if too large
    max_neurons = 20
    if n_out > max_neurons:
        # Select first max_neurons
        weights_array = weights_array[:, :max_neurons, :]
        n_out = max_neurons
    
    if n_in > 50:
        step = max(1, n_in // 50)
        weights_array = weights_array[:, :, ::step]
        n_in = weights_array.shape[2]
    
    global_min = np.min(weights_array)
    global_max = np.max(weights_array)
    
    print(f"Visualization shape: {weights_array.shape}")
    print(f"Weight range: [{global_min:.4f}, {global_max:.4f}]")
    
    # Create figure
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    x = np.arange(n_in)
    y = np.arange(n_out)
    X, Y = np.meshgrid(x, y)
    
    # Initial surface
    Z = np.zeros_like(X)
    surf = ax.plot_surface(X, Y, Z, 
                          cmap=cm.viridis,
                          vmin=global_min,
                          vmax=global_max,
                          rstride=1,
                          cstride=1)
    
    ax.set_xlabel('Input Connections', fontsize=12)
    ax.set_ylabel('Output Neurons', fontsize=12)
    ax.set_zlabel('Epoch', fontsize=12)
    ax.view_init(elev=25, azim=45)
    
    # Colorbar
    mappable = cm.ScalarMappable(norm=Normalize(vmin=global_min, vmax=global_max), cmap=cm.viridis)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=20, pad=0.1)
    
    ax.set_title('3D Weight Heatmap Over Training', fontsize=14)
    
    def update(frame):
        weights_2d = weight_history[frame]
        
        # Apply same subsampling
        if n_out > max_neurons:
            weights_2d = weights_2d[:max_neurons, :]
        if n_in > 50:
            weights_2d = weights_2d[:, ::step]
        
        Z = np.full_like(X, frame)
        surf._offsets3d = (X, Y, Z)
        surf._facecolors = cm.viridis((weights_2d - global_min) / (global_max - global_min + 1e-10))
        surf.changed()
        
        # Fix for numpy 1.26 compatibility
        weights_2d_arr = np.array(weights_2d)
        current_min = float(weights_2d_arr.min())
        current_max = float(weights_2d_arr.max())
        ax.set_title(f'Epoch {frame}: [{current_min:.4f}, {current_max:.4f}]', fontsize=12)
        
        return surf, ax
    
    ani = FuncAnimation(fig, update, frames=n_epochs, interval=1000/5, blit=False)
    
    # Save video
    writer = FFMpegWriter(fps=5, bitrate=10000)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Test video saved to: {output_path}")
    return output_path


def main():
    """Run the test."""
    print("="*60)
    print("Running 3D Visualization Test")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Test 3D Visualization')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--output', type=str, default="../../tmp/test_3d_small.mp4", help='Output path')
    args = parser.parse_args()
    
    # Setup
    config, fabric, train_loader, test_loader, feature_extractor, lateral_network = \
        setup_simple_training(epochs=args.epochs)
    
    # Run training and collect weights
    weight_history = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}...")
        weights = run_single_epoch(config, feature_extractor, lateral_network, train_loader, epoch)
        weight_history.append(weights)
        # Ensure weights is a numpy array
        weights_np = np.array(weights)
        w_min, w_max = float(np.min(weights_np)), float(np.max(weights_np))
        print(f"  Weights shape: {weights_np.shape}, range: [{w_min:.4f}, {w_max:.4f}]")
    
    # Create visualization
    create_test_visualization(weight_history, args.output)
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print(f"Output: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
