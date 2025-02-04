import os
import argparse
import laspy
import numpy as np
import matplotlib.pyplot as plt

CLASS_ID_2_NAME = {
    0: 'Created, never classified',
    1: 'Unclassified',
    2: 'Ground',
    3: 'Low Vegetation',
    4: 'Medium Vegetation',
    5: 'High Vegetation',
    6: 'Building',
    7: 'Low Point (noise)',
    8: 'Model Key-point (mass point)',
    9: 'Water',
    17: 'Bridge Deck',
    18: 'Noise'
}

def load_point_cloud(fn, format='npz'):
    if format == 'npz':
        data_npz = np.load(fn)
        points = data_npz['points']
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
        classification = np.array(laz_file.classification).astype(np.uint8)
        points = np.vstack([x, y, z, r, g, b, intensity, classification]).transpose()
        return points
    else:
        raise ValueError(f"Unknown format {format}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and plot point cloud intensity distributions.")
    parser.add_argument("--file_path", type=str,
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2019_ogc_fid_1_1_30m.npz",
                        help="Path to point cloud file.")
    parser.add_argument("--format", type=str, choices=["npz", "laz"], default="npz",
                        help="Point cloud file format (default: npz).")
    parser.add_argument("--output_dir", type=str,
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_ply/")
    args = parser.parse_args()

    file_path = args.file_path
    file_format = args.format
    output_dir = args.output_dir
    file_name = os.path.basename(file_path).replace(".npz", "").replace(".laz", "")

    if os.path.exists(file_path):
        points = load_point_cloud(file_path, file_format)

        # Plot intensity distribution for all points
        intensity_all = points[:, 6]
        plt.figure(figsize=(8, 6))
        plt.hist(intensity_all, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.title("Intensity Distribution for All Points")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_intensity_distribution_all_points.png"))
        plt.close()

        # Select ground points (class id == 2)
        ground_mask = points[:, 7] == 2
        if np.sum(ground_mask) > 0:
            intensity_ground = points[ground_mask][:, 6]
            plt.figure(figsize=(8, 6))
            plt.hist(intensity_ground, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.title("Intensity Distribution for Ground Points")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{file_name}_intensity_distribution_ground_points.png"))
            plt.close()
        else:
            print("No ground points (class id == 2) found in the data.")

        print("Plots saved as 'intensity_distribution_all_points.png' and 'intensity_distribution_ground_points.png'.")
    else:
        print(f"Warning: File {file_path} not found. Skipping...")
