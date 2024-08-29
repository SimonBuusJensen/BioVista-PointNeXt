from torch.utils.data import Dataset
from ..build import DATASETS
import os 
import pandas as pd
import numpy as np
import laspy
import torch


@DATASETS.register_module()
class BioVista(Dataset):

    num_classes = 2
    classes = ['low_biodiversity_potential_forest', 'high_biodiversity_potential_forest']
    gravity_dim = 2

    def __init__(self,
                 data_root='/workspace/dataset',
                 split='train',
                 num_points=8192,
                 transform=None
                 ):
        
        # Assert the data_root is a csv file
        assert data_root.endswith('.csv')
        csv_file = data_root
        self.df = pd.read_csv(csv_file)
        self.num_points = num_points

        self.shape_size = 30
        self.radius = int(self.shape_size / 2)

        self.data_root = os.path.dirname(csv_file)
        self.point_cloud_root = os.path.join(self.data_root, f"{self.shape_size}m_airborne_lidar_scanner")
        assert os.path.exists(self.point_cloud_root), f"Point cloud root {self.point_cloud_root} does not exist"
        assert len(os.listdir(self.point_cloud_root)) > 0, f"Point cloud root {self.point_cloud_root} is empty"

        self.split = split
        # assert split is either train or val
        assert split in ['train', 'test']

        self.df = self.df[self.df['dataset_split'] == split]

        self.transform = transform
    
    def apply_circle_mask(self, point_cloud_array, radius, center_x, center_y):
        """
        Method for applying a circular mask to the point cloud.
        1. Remove all the points outside the circle using the radius and the x- and y-coordinates
        """
        mask = (point_cloud_array[:, 0] - center_x)**2 + (point_cloud_array[:, 1] - center_y)**2 <= radius**2 # Use Pythagoras theorem to find the points inside the circle
        point_cloud_array = point_cloud_array[mask]
        return point_cloud_array

    def __len__(self):
        return len(self.df)

    def file_name_from_row(self, row):
        id = row["id"]
        ogc_fid = row["ogc_fid"]
        class_name = row["class_name"].replace(" ", "_").lower()
        fn = f"id_{id}_ogc_fid_{ogc_fid}_{class_name}_point_frac_1_30m.laz"
        return fn
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        fn = self.file_name_from_row(row)
        fn = os.path.join(self.point_cloud_root, fn)
        assert os.path.exists(fn), f"Point cloud file {fn} does not exist"

        # Load the x, y, z and intensity coordinates of the point cloud
        # Read LAS for given plot ID
        laz_file = laspy.read(fn)
     
        # Cast x, y and z to float32 (to save memory and speed up processing)
        x = np.array(laz_file.x).astype(np.float32)
        y = np.array(laz_file.y).astype(np.float32)
        z = np.array(laz_file.z).astype(np.float32)

        # Basic data points
        points = [x, y, z]
        # points = [x, y, z, laz_file.intensity]

        # Convert to N x C format
        points = np.vstack(points).transpose()

        # Identify center of the point cloud
        center_x, center_y = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2, (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
        points = self.apply_circle_mask(points, self.radius, center_x, center_y)

        # Select a random subset of points
        if points.shape[0] > self.num_points:
            idx = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[idx, :]

        # Fill more points if the number of points is less than the required number of points
        if points.shape[0] < self.num_points:
            n_points_to_fill = self.num_points - points.shape[0]
            idx = np.random.choice(points.shape[0], n_points_to_fill, replace=True)
            points = np.concatenate([points, points[idx, :]], axis=0)
            # print(f"Filling {n_points_to_fill} extra points into {fn}")
    
        data = {
            'pos': points,
            'y': row["class_id"]
        }

        if self.transform is not None:
            data = self.transform(data)

        data['x'] = torch.cat((data['pos'], torch.from_numpy(points[:, self.gravity_dim:self.gravity_dim+1] - points[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)

        return fn, data