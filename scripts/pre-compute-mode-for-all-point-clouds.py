import os
import laspy
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Script for computing the mode for ground points for all point clouds in the BioVista dataset
"""

def load_point_cloud(fn, format='npz'):
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
        points = data_npz['points']

        # Keep only the x, y, z, intensity and class_id columns (channel 0, 1, 2, 6, 7)
        points = points[:, [0, 1, 2, 6, 7]]

        return points
    elif format == 'laz':
        laz_file = laspy.read(fn)
        x = np.array(laz_file.x).astype(np.float32)
        y = np.array(laz_file.y).astype(np.float32)
        z = np.array(laz_file.z).astype(np.float32)
        intensity = laz_file.intensity
        laz_class_ids = laz_file.classification

        # Basic data points, x, y, z, intensity and laz class id (not to be confused with the class id in the dataset)
        points = [x, y, z, intensity, laz_class_ids]

        # Convert to N x C format e.g. 1000 X 5 where 5 is x, y, z, intensity and laz class id
        points = np.vstack(points).transpose()
        return points

def file_name_from_row(row, format='npz'):
        id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        year = row["year"]
        class_name = row["class_name"].replace(" ", "_").lower()
        if format == 'npz':
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.npz"
        else:
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.laz"
        return fn


if __name__ == "__main__":

    sample_csv = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv"
    df = pd.read_csv(sample_csv)
    point_cloud_format = 'npz'

    if point_cloud_format == 'laz':
        point_cloud_root = os.path.join(os.path.dirname(sample_csv), f"ALS_point_clouds")
    else:
        point_cloud_root = os.path.join(os.path.dirname(sample_csv), f"ALS_point_clouds_npz")

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        fn = file_name_from_row(row, point_cloud_format)
        fp = os.path.join(point_cloud_root, fn)

        assert os.path.exists(fp), f"File {fp} does not exist"

        points = load_point_cloud(fp, format='npz')

        mask = (points[:, 4] == 2) | (points[:, 4] == 3)
        ground_intensity_array = points[mask][:, 3]

        # Calculate the mode of the intensity values using np.hist
        counts, bins = np.histogram(ground_intensity_array, bins=50)
        mode = bins[np.argmax(counts)]
        if mode == 0:
            # If the mode is 0 we use the second most common value, as 0 will cause division by zero errors if with_normalize_intensity is True
            print(f"Mode is 0 in file {fn}")
            mode = bins[np.argsort(counts)[-2]]
            print("Using second most common value as mode: ", mode)

        df.at[i, 'mode'] = mode
        

    # Save the df with the mode values
    df.to_csv(sample_csv, index=False)
