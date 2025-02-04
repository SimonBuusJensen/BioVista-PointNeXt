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

def process_csv(input_csv, point_cloud_dir, output_csv, file_format):
    """
    Process each row in the input CSV file, load corresponding point cloud,
    compute statistics, and save to an output CSV file.
    """
    df = pd.read_csv(input_csv)

    # Sample only the rows where dataset_split is 'val'
    df = df[df["dataset_split"] == "train"]

    with open(output_csv, "w") as f:

        write_header = True  # Ensure header is written only once
        
        for row_idx, (_, row) in enumerate(df.iterrows()):
            
            # Print progress every 100 rows
            if (row_idx + 1) % 100 == 0:
                print(f"Processing row {row_idx + 1}/{len(df)}")

            file_name = file_name_from_row(row, file_format)
            file_path = os.path.join(point_cloud_dir, file_name)
            
            if os.path.exists(file_path):
                points = load_point_cloud(file_path, file_format)

                stats = {}
                stats["file_name"] = file_name
                stats["class_name"] = row["class_name"].replace(" ", "_").lower()
                stats["year"] = row["year"]
                stats["ogc_fid"] = row["ogc_fid"]
                stats["sample_id"] = row["sample_id"]
                stats["dataset_split"] = row["dataset_split"]

                stats["num_points"] = points.shape[0]
                stats["intensity_mean"] = np.mean(points[:, 6])
                stats["intensity_std"] = np.std(points[:, 6])

                # Get the intensity statistics for each class ['Ground', 'Low Vegetation', 'Medium Vegetation', 'High Vegetation']
                # 2: 'Ground'
                class_points = points[points[:, 7] == 2]
                if class_points.shape[0] > 0:

                    intensity_mean = np.mean(class_points[:, 6])
                    intensity_std = np.std(class_points[:, 6])

                    stats["num_points_ground"] = class_points.shape[0] 
                    stats["intensity_mean_ground"] = intensity_mean 
                    stats["intensity_std_ground"] = intensity_std
                else:
                    stats["num_points_ground"] = 0
                    stats["intensity_mean_ground"] = 0
                    stats["intensity_std_ground"] = 0
                
                # 3: 'Low Vegetation'
                class_points = points[points[:, 7] == 3]
                if class_points.shape[0] > 0:

                    intensity_mean = np.mean(class_points[:, 6])
                    intensity_std = np.std(class_points[:, 6])

                    stats["num_points_low_vegetation"] = class_points.shape[0] 
                    stats["intensity_mean_low_vegetation"] = intensity_mean 
                    stats["intensity_std_low_vegetation"] = intensity_std
                else:
                    stats["num_points_low_vegetation"] = 0
                    stats["intensity_mean_low_vegetation"] = 0
                    stats["intensity_std_low_vegetation"] = 0

                # 4: 'Medium Vegetation'
                class_points = points[points[:, 7] == 4]
                if class_points.shape[0] > 0:

                    intensity_mean = np.mean(class_points[:, 6])
                    intensity_std = np.std(class_points[:, 6])

                    stats["num_points_medium_vegetation"] = class_points.shape[0] 
                    stats["intensity_mean_medium_vegetation"] = intensity_mean 
                    stats["intensity_std_medium_vegetation"] = intensity_std
                else:
                    stats["num_points_medium_vegetation"] = 0
                    stats["intensity_mean_medium_vegetation"] = 0
                    stats["intensity_std_medium_vegetation"] = 0

                #5: 'High Vegetation'
                class_points = points[points[:, 7] == 5]
                if class_points.shape[0] > 0:

                    intensity_mean = np.mean(class_points[:, 6])
                    intensity_std = np.std(class_points[:, 6])

                    stats["num_points_high_vegetation"] = class_points.shape[0]  
                    stats["intensity_mean_high_vegetation"] = intensity_mean 
                    stats["intensity_std_high_vegetation"] = intensity_std
                else:
                    stats["num_points_high_vegetation"] = 0
                    stats["intensity_mean_high_vegetation"] = 0
                    stats["intensity_std_high_vegetation"] = 0
        
                # Append to CSV instead of keeping in memory
                pd.DataFrame([stats]).to_csv(f, mode='a', header=write_header, index=False)
                write_header = False  # Only write the header in the first row
            else:
                print(f"Warning: File {file_name} not found. Skipping...")

    # Convert results to DataFrame and save to CSV
    print(f"Processing completed. Results saved to {output_csv}")

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
    if format == "npz":
        point_cloud_dir = os.path.join(os.path.dirname(input_csv), "ALS_point_clouds_npz")
    else:
        point_cloud_dir = os.path.join(os.path.dirname(input_csv), "ALS_point_clouds")

    process_csv(input_csv, point_cloud_dir, output_csv, format)

