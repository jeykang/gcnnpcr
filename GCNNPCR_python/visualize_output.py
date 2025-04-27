import torch
import open3d as o3d
import numpy as np
import os

def create_point_cloud(points, colors=None):
    """Create an Open3D point cloud from numpy array."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_results(filepath):
    """
    Load and visualize the saved LiDAR data and model output.
    Args:
        filepath: Path to the saved .pt file
    """
    # Load the saved data
    data = torch.load(filepath)
    input_data = data['input']  # [N,6] tensor (xyz + normals)
    output = data['output']     # Model output
    center = data['center']     # Original center for denormalization
    scale = data['scale']       # Original scale for denormalization

    # Convert input points back to world space
    input_xyz = input_data[:, :3].numpy() * scale + center
    
    # If output is a dictionary (depends on your model), adjust accordingly
    if isinstance(output, dict):
        output_xyz = output['points'].squeeze(0).detach().numpy() * scale + center
    else:
        output_xyz = output.squeeze(0).detach().numpy() * scale + center

    # Create colored point clouds
    input_pcd = create_point_cloud(
        input_xyz,
        colors=np.tile([1, 0, 0], (len(input_xyz), 1))  # Red for input
    )
    
    output_pcd = create_point_cloud(
        output_xyz,
        colors=np.tile([0, 1, 0], (len(output_xyz), 1))  # Green for output
    )

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    # Visualize
    o3d.visualization.draw_geometries([
        input_pcd,    # Input points (red)
        output_pcd,   # Output points (green)
        coord_frame   # Coordinate frame
    ])

if __name__ == "__main__":
    # Example usage
    result_path = "path/to/your/lidar_model_output.pt"
    if os.path.exists(result_path):
        visualize_results(result_path)
    else:
        print(f"Could not find result file: {result_path}")
