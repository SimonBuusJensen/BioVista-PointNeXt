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

def create_heatmap(num_colors=200):
    # Define the color gradients
    colors = np.array([
        [0, 0, 255],    # Blue
        [75, 0, 180],   # Purple to transition smoothly
        [150, 0, 105],  # More towards red
        [225, 0, 30],   # Almost red
        [255, 0, 0]     # Red
    ])
    
    # Number of points in the heatmap
    heatmap = np.zeros((num_colors, 3), dtype=int)
    
    # Define the breakpoints for the color gradient transitions
    breakpoints = np.linspace(0, num_colors, len(colors))
    
    for i in range(1, len(colors)):
        # Calculate the range of indices for the current gradient
        start_idx = int(breakpoints[i-1])
        end_idx = int(breakpoints[i])
        num_indices = end_idx - start_idx
        
        # Linearly interpolate between the two colors over the indices
        for j in range(num_indices):
            weight = j / num_indices
            heatmap[start_idx + j] = (colors[i-1] * (1 - weight) + colors[i] * weight).astype(int)
    
    return heatmap

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and plot point cloud intensity distributions.")
    parser.add_argument("--file_path", type=str,
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2019_ogc_fid_32_157_30m.npz",
                        help="Path to point cloud file.")
    parser.add_argument("--format", type=str, choices=["npz", "laz"], default="npz",
                        help="Point cloud file format (default: npz).")
    parser.add_argument("--output_dir", type=str,
                        default="/home/simon/Desktop/cropped/")
    args = parser.parse_args()

    file_path = args.file_path
    file_format = args.format
    output_dir = args.output_dir
    file_name = os.path.basename(file_path).replace(".npz", "").replace(".laz", "")

    ground_color = '#b3a47c'
    # green_color_2 = '#61CD57'

    cmap = create_heatmap(num_colors=256)

    n_bins = 50
    if os.path.exists(file_path):
        points = load_point_cloud(file_path, file_format)

        # Plot intensity distribution for all points
        intensity_all = points[:, 6]
        plt.figure(figsize=(8, 6))

        counts, bin_edges, patches = plt.hist(intensity_all, bins=n_bins, edgecolor='black', alpha=0.7)

        # Use the bin_edges to determine the color of each patch
        color_array = cmap / 255.0

        for bin_edge, patch in zip(bin_edges, patches):
            idx = int((bin_edge - np.min(bin_edges)) / (np.max(bin_edges) - np.min(bin_edges)) * 255)
            patch.set_facecolor(color_array[idx])
        
        # plt.hist(intensity_all, bins=n_bins, color=green_color_2, edgecolor='black', alpha=0.7)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # plt.title("Intensity Distribution for All Points")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_name}_intensity_distribution_all_points.png"), dpi=1000)
        plt.close()

        # Select ground points (class id == 2)
        ground_mask = points[:, 7] == 2
        if np.sum(ground_mask) > 0:
            intensity_ground = points[ground_mask][:, 6]
            plt.figure(figsize=(8, 6))

            counts, bin_edges, patches = plt.hist(intensity_ground, bins=n_bins, edgecolor='black', alpha=0.7)

            # Use the bin_edges to determine the color of each patch 
            color_array = cmap / 255.0
 
            for bin_edge, patch in zip(bin_edges, patches):
                idx = int((bin_edge - np.min(bin_edges)) / (np.max(bin_edges) - np.min(bin_edges)) * 255)
                patch.set_facecolor(color_array[idx])

            # Set a vertical line at the mode
            counts, bins = np.histogram(intensity_ground, bins=n_bins)
            mode = bins[np.argmax(counts)]
            print(mode)
            plt.axvline(mode, color='black', linestyle='dashed', linewidth=2)


            # plt.hist(intensity_ground, bins=n_bins, color=ground_color, edgecolor='black', alpha=0.7)
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            # plt.title("Intensity Distribution for Ground Points")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()


            plt.savefig(os.path.join(output_dir, f"{file_name}_intensity_distribution_ground_points.png"), dpi=1000)
            plt.close()
        else:
            print("No ground points (class id == 2) found in the data.")

        print("Plots saved as 'intensity_distribution_all_points.png' and 'intensity_distribution_ground_points.png'.")
    else:
        print(f"Warning: File {file_path} not found. Skipping...")
