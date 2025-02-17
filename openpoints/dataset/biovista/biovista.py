from torch.utils.data import Dataset
from ..build import DATASETS
import os
import logging
import pandas as pd
import numpy as np
import laspy
import torch


@DATASETS.register_module()
class BioVista(Dataset):

    num_classes = 2
    classes = ['low_bio', 'high_bio']
    gravity_dim = 2

    def __init__(self,
                 data_root='/workspace/dataset',
                 split='train',
                 num_points=8192,
                 channels="xyzi",
                 transform=None,
                 normalize_intensity=False,
                 normalize_intensity_scale=1.0,
                 format='npz',
                 seed=None,
                 ):

        # Assert the data_root is a csv file
        assert data_root.endswith('.csv')
        csv_file = data_root
        self.df = pd.read_csv(csv_file)

        if split == "test":
            if seed is None:
                seed = 42
            else:
                self.seed = seed

        self.num_points = num_points
        self.channels = channels
        self.n_channels = len(channels)
        if "i" in channels:
            self.normalize_intensity = normalize_intensity
            self.normalize_intensity_scale = normalize_intensity_scale

        self.shape_size = 30
        self.radius = int(self.shape_size / 2)

        self.data_root = os.path.dirname(csv_file)

        self.format = format
        assert self.format in ['npz', 'laz']
        if self.format == 'laz':
            self.point_cloud_root = os.path.join(self.data_root, f"ALS_point_clouds")
        else:
            self.point_cloud_root = os.path.join(self.data_root, f"ALS_point_clouds_npz")

        assert os.path.exists(self.point_cloud_root), f"Point cloud root {self.point_cloud_root} does not exist"
        assert len(os.listdir(self.point_cloud_root)) > 0, f"Point cloud root {self.point_cloud_root} is empty"

        self.split = split
        # assert split is either train or val
        assert split in ['train', 'val', 'test']

        self.df = self.df[self.df['dataset_split'] == split]
        # Shuffle the dataframe
        if split == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.transform = transform

    def apply_circle_mask(self, point_cloud_array, radius, center_x, center_y):
        """
        Method for applying a circular mask to the point cloud.
        1. Remove all the points outside the circle using the radius and the x- and y-coordinates
        """
        mask = (point_cloud_array[:, 0] - center_x)**2 + (point_cloud_array[:, 1] -
                                                          center_y)**2 <= radius**2  # Use Pythagoras theorem to find the points inside the circle
        point_cloud_array = point_cloud_array[mask]
        return point_cloud_array

    def file_name_from_row(self, row):
        id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        year = row["year"]
        class_name = row["class_name"].replace(" ", "_").lower()
        if self.format == 'npz':
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.npz"
        else:
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.laz"
        return fn

    def load_point_cloud(self, fn):
        if self.format == 'npz':
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
        elif self.format == 'laz':
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

    def getitem_by_class_id_year_ogc_fid_and_sample_id(self, class_id, year, ogc_fid, sample_id):
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if row['class_id'] == class_id and row['year'] == year and row['ogc_fid'] == ogc_fid and row['sample_id'] == sample_id:
                idx = i
                break
           
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = self.file_name_from_row(row)
        fn = os.path.join(self.point_cloud_root, fn)

        try:
            assert os.path.exists(fn), f"Point cloud file {fn} does not exist"
            points = self.load_point_cloud(fn)

            # Remove the points which belong to class 7 (noise) or 18 (noise)
            points = points[(points[:, 4] != 7) & (points[:, 4] != 18)]

            # Remove points which for whatever reason have a negative z values (below the ground)
            points = points[points[:, 2] > 0]

            if "i" in self.channels and self.normalize_intensity:
                try:
                    mode = round(row["mode"], 2)
                except:
                    logging.info(f"Mode not found in file {fn}. Calculating mode from the intensity values")
                    mask = (points[:, 4] == 2) | (points[:, 4] == 3)
                    ground_intnsity_array = points[mask][:, 3]

                    # Calculate the mode of the intensity values using np.hist
                    counts, bins = np.histogram(ground_intnsity_array, bins=50)
                    mode = bins[np.argmax(counts)]
                    if mode == 0:
                        # If the mode is 0 we use the second most common value, as 0 will cause division by zero errors if with_normalize_intensity is True
                        mode = bins[np.argsort(counts)[-2]]
                        assert mode != 0, f"Second most common value is 0 in file {fn}"
                        logging.info(f"Mode is 0 in file {fn}. Using second most common value as mode: ", mode)

            # Identify center of the point cloud and apply a circular mask using Pythagoras theorem
            center_x, center_y = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2, (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
            points = self.apply_circle_mask(points, self.radius, center_x, center_y)

            if self.split == 'test':
                np.random.seed(self.seed)
            
            # Select a random subset of points given the self.num_points
            if self.num_points is not None and points.shape[0] >= self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices, :]
                if idx == 0 or idx == 25 or idx == 49:
                    print("idx: ", idx)
                    print(indices[:10])

            # Fill extra points into the points cloud if the number of points is less than the required number of points
            if self.num_points is not None and points.shape[0] < self.num_points:
                n_points_to_fill = self.num_points - points.shape[0]
                indices = np.random.choice(points.shape[0], n_points_to_fill, replace=True)
                points = np.concatenate([points, points[indices, :]], axis=0)
                if idx == 0 or idx == 25 or idx == 49:
                    print("idx: ", idx)
                    print(indices[:10])

            data = {
                'pos': points[:, :3],
                'y': row["class_id"]
            }

            # Apply the transform to the data
            # train: [PointsToTensor, PointCloudScaling, PointCloudXYZAlign, PointCloudRotation, PointCloudJitter]
            # val: [PointsToTensor, PointCloudXYZAlign]
            if self.transform is not None:
                data = self.transform(data)

            data['x'] = data['pos']

            # Append the gravity dimension to the data (height of the point cloud)
            if "h" in self.channels:
                point_heights = points[:, self.gravity_dim:self.gravity_dim +
                                     1] - points[:, self.gravity_dim:self.gravity_dim+1].min()
                point_heights_tensor = torch.from_numpy(point_heights) 
                data['x'] = torch.cat((data['x'], point_heights_tensor), dim=1)

            # Append the intensity dimension to the data
            if "i" in self.channels:
                intensity_array = points[:, 3]
            
                if self.normalize_intensity:
                    if self.normalize_intensity_scale == 1.0:
                        intensity_array = (intensity_array / mode)
                    else:
                        intensity_array = (intensity_array / mode) * self.normalize_intensity_scale
                
                # Check for nans in the intensity array
                if np.isnan(intensity_array).any():
                    logging.info(f"Intensity array has nans in file {fn}")

                intensity_tensor = torch.from_numpy(intensity_array[:, np.newaxis])
                
                data['x'] = torch.cat((data['x'], intensity_tensor), dim=1)

            # Assert the data['x'] has the correct shape N x C=5
            assert data['x'].shape[1] == self.n_channels, f"Data['x'] has shape {data['x'].shape} instead of N x C={self.n_channels}"

            return fn, data
        
        except Exception as e:
            logging.info(f"Error: {e} in file {fn}")
            return self.__getitem__(idx + 1)

    def __len__(self):
        return len(self.df)
