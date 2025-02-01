from torch.utils.data import Dataset
from pathlib import Path
import os 
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import laspy
import open3d as o3d
import matplotlib.pyplot as plt 
import torch
from torchvision.transforms import Compose
from openpoints.transforms import PointsToTensor, PointCloudXYZAlign, PointCloudRotation, PointCloudJitter
from PIL import Image

center = None

class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        
        # TODO: Delete this again:
        # Set the scale to the max for all axes
        scale = torch.ones(3, dtype=torch.float32, device=device) * self.scale_max

        # Set the scale to the min for all axes

        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: 
                scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data

def apply_circle_mask(point_cloud_array, radius, center_x, center_y):
    """
    Method for applying a circular mask to the point cloud.
    1. Remove all the points outside the circle using the radius and the x- and y-coordinates
    """
    mask = (point_cloud_array[:, 0] - center_x)**2 + (point_cloud_array[:, 1] - center_y)**2 <= radius**2 # Use Pythagoras theorem to find the points inside the circle
    point_cloud_array = point_cloud_array[mask]
    return point_cloud_array


def load_point_cloud(fn, format = 'npz'):
    if format == 'npz':
        """
        0: x (float32), 
        1: y (float32),
        2: z (float32),
        3: r (uint8),
        4: g (uint8),
        5: b (uint8),
        6: intensity float32,
        7: class_id uint8
        """
        data_npz = np.load(fn)
        points = data_npz['points'] # 
        return points
    
    elif format == 'laz':
        laz_file = laspy.read(fn)
        x = np.array(laz_file.x).astype(np.float32)
        y = np.array(laz_file.y).astype(np.float32)
        z = np.array(laz_file.z).astype(np.float32)
        r = np.array(laz_file.red).astype(np.uint8)
        g = np.array(laz_file.green).astype(np.uint8)
        b = np.array(laz_file.blue).astype(np.uint8)
        intensity = np.array(laz_file.intensity).astype(np.float32)
        laz_class_ids = np.array(laz_file.classification).astype(np.uint8)

        # Basic data points, x, y, z, intensity and laz class id (not to be confused with the class id in the dataset)
        points = [x, y, z, r, g, b, intensity, laz_class_ids]

        # Convert to N x C format e.g. 1000 X 5 where 5 is x, y, z, intensity and laz class id
        points = np.vstack(points).transpose()
        return points
    else:
        raise ValueError(f"Unknown format {format}")

def create_colormap(num_colors=400):

    # Create a colormap using matplotlib
    cmap = plt.get_cmap('jet')

    # Generate colors for each index
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    # Convert to 8-bit RGB format
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

    return colors_rgb
    
def points_2_ply( points, fn, color_mode='rgb', n_colors=256):
    # Save the points as a ply file (x, y, z, r, g, b)

    num_points = points.shape[0]

    # Create a directory for the file if it doesn't exist
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)

    if color_mode == 'intensity':
        max_intensity = np.max(points[:, 6])
        min_intensity = np.min(points[:, 6])
        print(f"Max intensity: {max_intensity}, Min intensity: {min_intensity}")
        # Normalize the intensity values to 0-n_colors
        normalized_intensities = (points[:, 6] - min_intensity) / (max_intensity - min_intensity) * (n_colors - 1)
        cmap = create_colormap(n_colors)
    
    if color_mode == 'height':
        min_height = np.min(points[:, 2])
        # Subtract the min height from the height values
        normalized_heights = points[:, 2] - min_height
        cmap = create_colormap(60)

    with open(fn, 'w') as file:
    # Writing the PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {num_points}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")   # 8-bit color
        file.write("property uchar green\n") # 8-bit color
        file.write("property uchar blue\n")  # 8-bit color
        file.write("end_header\n")

        # Writing the point data
        for i in range(num_points):
            x, y, z, r, g, b, *_ = points[i]  # Use * to ignore extra columns

            if color_mode == 'rgb':
                file.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            elif color_mode == 'intensity':
                color = cmap[int(normalized_intensities[i])]
                file.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")
            elif color_mode == 'height':
                color = cmap[int(normalized_heights[i])]
                file.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")

        file.close()

def open_ply_file(filename):

    with open(filename, 'r') as file:
        # Skip the header
        while True:
            line = file.readline()
            if line.startswith("end_header"):
                break
        
        # Read points data
        points_data = []
        for line in file:
            parts = line.split()
            # if len(parts) < 7:  # Ensure there are enough values in the line
            #     continue
            x, y, z, r, g, b = map(float, parts[:6])
            points_data.append((float(x), float(y), float(z), int(r), int(g), int(b)))  # Convert to integers

    return np.array(points_data)

def capture_view(vis, view_params, save_path):
    """Captures a view of the point cloud with given camera parameters."""
    view_control = vis.get_view_control()
    view_control.set_front(view_params['front'])
    view_control.set_lookat(view_params['lookat'])
    view_control.set_up(view_params['up'])
    view_control.set_zoom(view_params['zoom'])
    
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)

def process_and_save_image(image_path):
    """Processes the PNG image and saves it in place."""
    img = process_png(image_path)  # Assuming process_png() is a function that processes the image
    img.save(image_path)

def ply_2_png(ply_data, save_name, calculate_center=True):
    global center
    """Converts a point cloud (PLY data) to multiple PNG views."""
    # Initialize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ply_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(ply_data[:, 3:6] / 255)

    # Create a directory for output images
    Path(os.path.dirname(save_name)).mkdir(parents=True, exist_ok=True)

    # Initialize visualizer only once
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Prevent GUI window pop-up
    vis.add_geometry(pcd)

    # Optimize render settings
    render_options = vis.get_render_option()
    render_options.point_color_option = o3d.visualization.PointColorOption.Color
    render_options.point_size = 6.0
    render_options.background_color = np.array([255, 255, 255]) / 255  # Normalize to [0,1]

    # Compute bounding box center
    if calculate_center or center is None:
        center = pcd.get_axis_aligned_bounding_box().get_center()
        print("Re-calculated center: ", center)
    else:
        center = center
        print("Re-using center: ", center)

    # Define multiple views
    view_params_list = [
        {"name": "_side_view", "front": [0.0, -1, 0.5], "up": [0, 0, 1], "zoom": 1.0},
        {"name": "_top_view", "front": [0, -0.05, 1.2], "up": [0, 0, 1], "zoom": 1.0}
    ]

    # Process all views in a loop
    for view_params in view_params_list:
        save_path = save_name.replace(".png", f"{view_params['name']}.png")
        view_params["lookat"] = center  # Set the lookat dynamically
        capture_view(vis, view_params, save_path)
        process_and_save_image(save_path)

    vis.destroy_window()  # Clean up Open3D visualizer

def process_png(image_path, remove_background=True, crop=True):
    # Open the image
    img = Image.open(image_path).convert("RGBA")

    # Fetch image dimensions
    width, height = img.size

    # Initialize bounding box coordinates
    left = width
    top = height
    right = 0
    bottom = 0

    # Loop through the image pixels
    for x in range(width):
        for y in range(height):
            r, g, b, a = img.getpixel((x, y))
            # Check if the pixel is white
            if r > 225 and g > 225 and b > 225:
                # Set alpha to zero
                if remove_background:
                    img.putpixel((x, y), (r, g, b, 0))
                else:
                    img.putpixel((x, y), (r, g, b, 255))
            
            else:
                # Update bounding box
                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y
    
    # Crop the image to the bounding box
    if crop:
        img = img.crop((left, top, right, bottom))

    return img      

def save_points(points, fn, color_mode, calculate_center=True):
    points_2_ply(points, fn, color_mode=color_mode)
    ply_point_cloud = open_ply_file(fn)
    # Save as a png
    ply_2_png(ply_point_cloud, fn.replace(".ply", ".png"), calculate_center=calculate_center)

if __name__ == "__main__":


    # fn = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/low_biodiversity_forest_2023_ogc_fid_18_3_30m.npz"
    fn = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2019_ogc_fid_1_6_30m.npz"
    # fn = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/low_biodiversity_forest_2020_ogc_fid_11__30m.npz"
    channels = 'xyzih'
    format = 'npz'
    color_mode = 'intensity' # rgb, intensity, height
    radius = 15
    num_points = 8192
    gravity_dim = 2
    is_save_points = True

    # Save dir for the ply files
    save_dir = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/figures/point_cloud_processing_pipeline/high_biodiversity_forest_2019_ogc_fid_1_6_30m"
    assert os.path.exists(fn), f"Point cloud file {fn} does not exist"

    points = load_point_cloud(fn) # x, y, z, r, g, b, intensity, class_id (laz class id)
    print(f"Number of points in point cloud: {points.shape}")
    # Print the min, max, mean and std of the x, y, z and intensity values
    print("Min x, y, z, r, g, b, intensity, class_id: ", points.min(axis=0))
    print("Max x, y, z, r, g, b, intensity, class_id: ", points.max(axis=0))
    print("Mean x, y, z, r, g, b, intensity, class_id: ", points.mean(axis=0))

    # Save the points as a ply file (x, y, z, r, g, b)
    fn_after_load = os.path.join(save_dir, "1_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}.ply"))
    print("---------------------------------")
    print("Step 1: Load the point cloud")
    if is_save_points:
        save_points(points, fn_after_load, color_mode)

    # Remove the points which belong to class 7 (noise) or 18 (noise)
    print("---------------------------------")
    print("Step 2: Remove noise points")
    print(f"Before removing noise points: {points.shape}")
    points = points[(points[:, -1] != 7) & (points[:, -1] != 18)]
    print(f"After removing noise points: {points.shape}")
    
    # Remove points which for whatever reason have a negative x, y or z value
    print("---------------------------------")
    print("Step 3: Remove points with negative x, y or z value")
    print(f"Before removing negative x, y or z values: {points.shape}")
    points = points[points[:, 0] > 0]
    points = points[points[:, 1] > 0]
    points = points[points[:, 2] > 0]
    print(f"After removing negative x, y or z values: {points.shape}")

    # Identify center of the point cloud and apply a circular mask using Pythagoras theorem
    print("---------------------------------")
    print("Step 4: Apply circular mask")
    center_x, center_y = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2, (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    print("Center x: ", center_x)
    print("Center y: ", center_y)
    points = apply_circle_mask(points, radius, center_x, center_y)
    print("Min x, y, z, r, g, b, intensity, class_id: ", points.min(axis=0))
    print("Max x, y, z, r, g, b, intensity, class_id: ", points.max(axis=0))
    print("Mean x, y, z, r, g, b, intensity, class_id: ", points.mean(axis=0))
    fn_after_circle_mask = os.path.join(save_dir, "4_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}_circle_mask.ply"))
    if is_save_points:
        save_points(points, fn_after_circle_mask, color_mode, calculate_center=False)

    # Select a random subset of points given the num_points
    if num_points is not None and points.shape[0] >= num_points:
        print("---------------------------------")
        print("Step 5: Select a random subset of points")
        print(f"Number of points before random subset: {points.shape}")
        idx = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[idx, :]
        fn_after_random_subset = os.path.join(save_dir, "5_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}_random_subset.ply"))
        print("Number of points after random subset: ", points.shape)
        print("Min x, y, z, r, g, b, intensity, class_id: ", points.min(axis=0))
        print("Max x, y, z, r, g, b, intensity, class_id: ", points.max(axis=0))
        print("Mean x, y, z, r, g, b, intensity, class_id: ", points.mean(axis=0))
        if is_save_points:
            save_points(points, fn_after_random_subset, color_mode, calculate_center=False)

    # Fill extra points into the points cloud if the number of points is less than the required number of points
    if num_points is not None and points.shape[0] < num_points:
        n_points_to_fill = num_points - points.shape[0]
        idx = np.random.choice(points.shape[0], n_points_to_fill, replace=True)
        points = np.concatenate([points, points[idx, :]], axis=0)

    data = {
        'pos': points[:,:3]
    }

    # Apply the transform to the data 1 by 1
    data = PointsToTensor()(data)
    
    print("---------------------------------")
    print("Step 6: Apply the random xyz align transform to the data")
    data = PointCloudXYZAlign(gravity_dim=2, normalize_gravity_dim=False)(data)
    # Convert the data['pos'] to numpy
    xyz_array = data['pos'].numpy()
    print("Number of points after xyzalign: ", xyz_array.shape)
    print("Min x, y, and z value after xyzalign: ", xyz_array.min(axis=0))
    print("Max x, y, and z value after xyzalign: ", xyz_array.max(axis=0))
    print("Mean x, y, and z value after xyzalign: ", xyz_array.mean(axis=0))
    points = np.concatenate([xyz_array, points[:, 3:]], axis=1)
    fn_after_xyz_align = os.path.join(save_dir, "6_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}_xyz_align.ply"))
    if is_save_points:
        save_points(points, fn_after_xyz_align, color_mode)

    print("---------------------------------")
    print("Step 7: Apply the scaling transform to the data")
    xyz_array = data['pos'].numpy()
    print("x, y, z length in meteres before scaling: ", xyz_array.max(axis=0) - xyz_array.min(axis=0))
    print("Mean x, y, and z value before scaling: ", np.round(xyz_array.mean(axis=0), 2))
    data = PointCloudScaling(scale=[0.9, 1.1])(data)
    xyz_array = data['pos'].numpy()
    print("Number of points after scaling: ", xyz_array.shape)
    print("x, y, z length in meteres after scaling: ", xyz_array.max(axis=0) - xyz_array.min(axis=0))
    print("Min x, y, and z value after scaling: ", xyz_array.min(axis=0))
    print("Max x, y, and z value after scaling: ", xyz_array.max(axis=0))
    print("Mean x, y, and z value after scaling: ", xyz_array.mean(axis=0))
    points = np.concatenate([xyz_array, points[:, 3:]], axis=1)
    fn_after_scaling = os.path.join(save_dir, "7_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}_scaling.ply"))
    if is_save_points:
        save_points(points, fn_after_scaling, color_mode, calculate_center=False)

    print("---------------------------------")
    print("Step 8: Apply the random rotation transform to the data")
    data = PointCloudRotation(angle=[0, 0, 1])(data)
    xyz_array = data['pos'].numpy()
    points = np.concatenate([xyz_array, points[:, 3:]], axis=1)
    fn_after_rotation = os.path.join(save_dir, "8_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}_rotation.ply"))
    if is_save_points:
        save_points(points, fn_after_rotation, color_mode, calculate_center=False)
    
    print("Number of points after rotations: ", points.shape)
    print("Min x, y, and z value after rotations: ", xyz_array.min(axis=0))
    print("Max x, y, and z value after rotations: ", xyz_array.max(axis=0))
    print("Mean x, y, and z value after rotations: ", xyz_array.mean(axis=0))
    
    print("---------------------------------")
    print("Step 9: Apply the random jitter transform to the data")
    data = PointCloudJitter(jitter_clip=0.02, jitter_sigma=0.005)(data)
    xyz_array = data['pos'].numpy()
    points = np.concatenate([xyz_array, points[:, 3:]], axis=1)
    fn_after_jitter = os.path.join(save_dir, "9_" + os.path.basename(fn).replace(f".{format}", f"_{color_mode}_jitter.ply"))
    if is_save_points:
        save_points(points, fn_after_jitter, color_mode, calculate_center=False)
    
    print("Number of points after jitter: ", points.shape)
    print("Min x, y, and z value after jitter: ", xyz_array.min(axis=0))
    print("Max x, y, and z value after jitter: ", xyz_array.max(axis=0))
    print("Mean x, y, and z value after jitter: ", xyz_array.mean(axis=0))

    data['x'] = data['pos']
    
    # Append the gravity dimension to the data (height of the point cloud)
    if "h" in channels:
        point_heights = points[:, gravity_dim:gravity_dim+1] - points[:, gravity_dim:gravity_dim+1].min()
        point_heights_tensor = torch.from_numpy(point_heights) # 8192 x 1
        data['x'] = torch.cat((data['x'], point_heights_tensor), dim=1)

        # Test if the min, max, mean of the point_heights_tensor is the same as the z values in data['pos']
        print("Min, max, mean of the point_heights_tensor: ", point_heights_tensor.min(), point_heights_tensor.max(), point_heights_tensor.mean())
        print("Min, max, mean of the z values in data['pos']: ", data['pos'][:, 2].min(), data['pos'][:, 2].max(), data['pos'][:, 2].mean())

    # Append the intensity dimension to the data
    if "i" in channels:
        intensity_tensor = torch.from_numpy(points[:, 6][:, np.newaxis])
        data['x'] = torch.cat((data['x'], intensity_tensor), dim=1)

