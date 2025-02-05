import os
import random
import laspy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Script for plotting the intensity of the classes by year:
- Pick 5 random samples from 2019, 2020, 2021, 2022, 2023 of High- and Low Biodiversity Forest
- For each, plot the intensity value distribution for all points, for ground/low vegetation points and for high vegetation points.
- Add the mean, median and mode as lines on the plot. 
"""
def file_name_from_row(row, format="npz"):
    id = row["sample_id"]
    ogc_fid = row["ogc_fid"]
    year = row["year"]
    class_name = row["class_name"].replace(" ", "_").lower()
    if format == 'npz':
        fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.npz"
    else:
        fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.laz"
    return fn

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

    parser = argparse.ArgumentParser(description="Process point cloud files from CSV metadata.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file.",
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv")
    parser.add_argument("--output_dir", type=str,
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/figures/point_cloud_intensity/")
    parser.add_argument("--format", type=str, choices=["npz", "laz"], default="npz", help="Point cloud file format (default: npz).")

    args = parser.parse_args()
    if args.format == "npz":
        file_path = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/"
    else:
        file_path = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds/"

    df = pd.read_csv(args.input_csv)

    """
    class_name: High Biodiversity Forest and Low Biodiversity Forest
    year: 2019, 2020, 2021, 2022, 2023
    """
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    for class_name in ["high biodiversity forest", "low biodiversity forest"]:
        for year in [2019, 2020, 2021, 2022, 2023]:
            
            # Get 5 random samples where class_name and year match
            samples = df[(df["class_name"] == class_name) & (df["year"] == year)].sample(1)

            for _, row in samples.iterrows():

                file_name = file_name_from_row(row, args.format)
                print(file_name)

                points = load_point_cloud(os.path.join(file_path, file_name), args.format)

                # Plot intensity distribution for all points
                intensity_all = points[:, 6]
                plt.figure(figsize=(8, 6))
                plt.hist(intensity_all, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

                 # Add the mean, median and mode
                # plt.axvline(np.mean(intensity_all), color='r', linestyle='dashed', linewidth=1)
                # plt.axvline(np.median(intensity_all), color='g', linestyle='dashed', linewidth=1)
                # Bin the values into 50 values and find the mode
                counts, bins = np.histogram(intensity_all, bins=50)
                mode = bins[np.argmax(counts)]
                plt.axvline(mode, color='b', linestyle='dashed', linewidth=1)
                plt.xlabel("Intensity")
                plt.ylabel("Frequency")
                plt.title("Intensity Distribution for All Points")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{file_name}_intensity_distribution_all_points_before_calibration.png"))
                plt.close()

                # Plot the intensity distribution for ground/low vegetations points (class id == 2 or class_id == 3) 
                mask = (points[:, 7] == 2) | (points[:, 7] == 3)
                if np.sum(mask) > 0:
                    intensity_ground = points[mask][:, 6]
                    plt.figure(figsize=(8, 6))
                    plt.hist(intensity_ground, bins=25, color='lightgreen', edgecolor='black', alpha=0.7)

                    # # Add the mean
                    # plt.axvline(np.mean(intensity_ground), color='r', linestyle='dashed', linewidth=1)
                    
                    # # Add the median
                    # plt.axvline(np.median(intensity_ground), color='g', linestyle='dashed', linewidth=1)

                    # Bin the values into 50 values and find the mode
                    counts, bins = np.histogram(intensity_ground, bins=50)
                    mode = bins[np.argmax(counts)]
                    print(mode)
                    plt.axvline(mode, color='b', linestyle='dashed', linewidth=1)

                    plt.xlabel("Intensity")
                    plt.ylabel("Frequency")
                    plt.title("Intensity Distribution for Ground/Low Vegetation Points")
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, f"{file_name}_intensity_distribution_ground.png"))
                    plt.close()


                # Plot the intensity distribution for high vegetation points (class id == 4)
                # mask = points[:, 7] == 4
                # if np.sum(mask) > 0:
                #     intensity_high_veg = points[mask][:, 6]
                #     plt.figure(figsize=(8, 6))
                #     plt.hist(intensity_high_veg, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)

                #     # Add the mean
                #     plt.axvline(np.mean(intensity_high_veg), color='r', linestyle='dashed', linewidth=1)
                    
                #     # Add the median
                #     plt.axvline(np.median(intensity_high_veg), color='g', linestyle='dashed', linewidth=1)

                #     # Bin the values into 50 values and find the mode
                #     counts, bins = np.histogram(intensity_high_veg, bins=50)
                #     mode = bins[np.argmax(counts)]
                #     plt.axvline(mode, color='b', linestyle='dashed', linewidth=1)

                #     plt.xlabel("Intensity")
                #     plt.ylabel("Frequency")
                #     plt.title("Intensity Distribution for High Vegetation Points")
                #     plt.tight_layout()
                #     plt.savefig(os.path.join(args.output_dir, f"{file_name}_intensity_distribution_high_veg.png"))
                #     plt.close()

                # Calibrate all points by dividing by the mode from the ground points
                intensity_all_calibrated = intensity_all / mode

                # Scale the range to be between 0 and 100
                plt.figure(figsize=(8, 6))
                plt.hist(intensity_all_calibrated, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                plt.xlabel("Intensity")
                plt.ylabel("Frequency")
                plt.title("Intensity Distribution for All Points After Calibration")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{file_name}_intensity_distribution_all_points_after_calibration.png"))
                plt.close()

                # Scale the range to be between 0 and 100
                intensity_all_calibrated = 100 * (intensity_all_calibrated - np.min(intensity_all_calibrated)) / (1.2 - np.min(intensity_all_calibrated))
                plt.figure(figsize=(8, 6))
                plt.hist(intensity_all_calibrated, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                plt.xlabel("Intensity")
                plt.ylabel("Frequency")
                plt.title("Intensity Distribution for All Points After Calibration")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{file_name}_intensity_distribution_all_points_after_calibration_scaled.png"))
                plt.close()

