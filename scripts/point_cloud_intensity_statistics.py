import os
import argparse
import laspy
import numpy as np
import pandas as pd

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



         
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process point cloud files from CSV metadata.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file.",
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv")
    parser.add_argument("--output_csv", type=str, help="Path to save the output CSV file.", 
                        default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/point_cloud_intensity_statistics.csv")
    parser.add_argument("--format", type=str, choices=["npz", "laz"], default="npz", help="Point cloud file format (default: npz).")

    args = parser.parse_args()

    input_csv = args.input_csv
    assert os.path.exists(input_csv), f"Input CSV file {input_csv} does not exist."

    output_csv = args.output_csv

    format = args.format
    assert format in ["npz", "laz"], f"Invalid format {format}. Choose from npz, laz."
    
    # file_path = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2019_ogc_fid_1_1_30m.npz"
    # file_path = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2021_ogc_fid_8_9_30m.npz"
    file_path = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/ALS_point_clouds_npz/high_biodiversity_forest_2022_ogc_fid_2_46_30m.npz"

    if os.path.exists(file_path):

        points = load_point_cloud(file_path, format)

        stats = {}
        fn = os.path.basename(file_path)
        # stats["file_name"] = fn
        # Derive class_name, year, ogc_fid, sample_id, dataset_split from the file name
        # stats["class_name"] = "_".join(fn.split("_")[:3])
        # stats["year"] = int(fn.split("_")[3])
        # stats["ogc_fid"] = int(fn.split("_")[6])
        # stats["sample_id"] = int(fn.split("_")[7])

        print("Number of points", points.shape[0])
        print("Min intensity", round(np.min(points[:, 6]), 0))
        print("Max intensity", round(np.max(points[:, 6]), 2))
        print("Mean intensity", round(np.mean(points[:, 6]), 2))
        print("Std intensity", round(np.std(points[:, 6]), 2))
        print()
    
        # Get the intensity statistics for each class ['Ground', 'Low Vegetation', 'Medium Vegetation', 'High Vegetation']
        # 2: 'Ground'
        class_points = points[(points[:, 7] == 2) | (points[:, 7] == 3)]
        if class_points.shape[0] > 0:
            print("num_points_ground", class_points.shape[0])
            print("Min intensity ground", round(np.min(class_points[:, 6]), 2))
            print("1st percentile intensity ground", round(np.percentile(class_points[:, 6], 1), 2))
            print("99th percentile intensity ground", round(np.percentile(class_points[:, 6], 99), 2))
            print("Max intensity ground", round(np.max(class_points[:, 6]), 2))
            print("Mean intensity_ground", round(np.mean(class_points[:, 6]), 2))
            print("Std intensity_ground", round(np.std(class_points[:, 6]), 2))
            print()

        # # 3: 'Low Vegetation'
        # class_points = points[points[:, 7] == 3]
        # if class_points.shape[0] > 0:
        #     print("num_points_low_vegetation", class_points.shape[0])
        #     print("Min intensity low vegetation", round(np.min(class_points[:, 6]), 2))
        #     print("Max intensity low vegetation", round(np.max(class_points[:, 6]), 2))
        #     print("Mean intensity low vegetation", round(np.mean(class_points[:, 6]), 2))
        #     print("Std intensity low vegetation", round(np.std(class_points[:, 6]), 2))
        #     print()
        
        # # 4: 'Medium Vegetation'
        # class_points = points[points[:, 7] == 4]
        # if class_points.shape[0] > 0:
        #     print("num_points_medium_vegetation", class_points.shape[0])
        #     print("Min intensity medium vegetation", round(np.min(class_points[:, 6]), 2))
        #     print("Max intensity medium vegetation", round(np.max(class_points[:, 6]), 2))
        #     print("Mean intensity medium vegetation", round(np.mean(class_points[:, 6]), 2))
        #     print("Std intensity medium vegetation", round(np.std(class_points[:, 6]), 2))
        #     print()
        
        # # 5: 'High Vegetation'
        # class_points = points[points[:, 7] == 5]
        # if class_points.shape[0] > 0:
        #     print("num_points_high_vegetation", class_points.shape[0])
        #     print("Min intensity high vegetation", round(np.min(class_points[:, 6]), 2))
        #     print("Max intensity high vegetation", round(np.max(class_points[:, 6]), 2))
        #     print("Mean intensity high vegetation", round(np.mean(class_points[:, 6]), 2))
        #     print("Std intensity high vegetation", round(np.std(class_points[:, 6]), 2))
        #     print()
        

        
        
    else:
        print(f"Warning: File {file_path} not found. Skipping...")

    # Convert results to DataFrame and save to CSV
    print(f"Processing completed. Results saved to {output_csv}")


