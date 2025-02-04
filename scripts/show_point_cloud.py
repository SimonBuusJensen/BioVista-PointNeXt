import laspy
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_colormap(num_colors=400):

    # Create a colormap using matplotlib
    cmap = plt.get_cmap('jet')

    # Generate colors for each index
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    # Convert to 8-bit RGB format
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

    return colors_rgb

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


if __name__ == "__main__":

    # fn = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/low_biodiversity_forest_2023_ogc_fid_18_3_30m.npz"
    fn = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2019_ogc_fid_1_6_30m.npz"
    color_mode = 'intensity' # rgb, intensity, height
    file_format = 'npz' # npz, laz

    points = load_point_cloud(fn, format = file_format)

    # Save the points as a ply file
    save_dir = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_ply/"
    ply_fn = os.path.join(save_dir, os.path.basename(fn).replace('.npz', '.ply'))
    points_2_ply(points, ply_fn, color_mode=color_mode)
    