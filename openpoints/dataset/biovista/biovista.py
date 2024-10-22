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
    classes = ['low_bio', 'high_bio']
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
        assert split in ['train', 'val', 'test']

        self.df = self.df[self.df['dataset_split'] == split]
        # Shuffle the dataframe
        if split == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        # Select randomly a subset of the dataframe 2000 train and 1000 val
        # if split == 'train':
        #     self.df = self.df.sample(2000)
        # else:
        #     self.df = self.df.sample(1000)
        # print("WARNING: Using a subset of the dataset")

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
        year = row["year"]
        class_name = row["class_name"].replace(" ", "_").lower()
        fn = f"id_{id}_ogc_fid_{ogc_fid}_{class_name}_year_{year}_30m.laz"
        # fn = f"id_{id}_ogc_fid_{ogc_fid}_{class_name}_point_frac_1_30m.laz"

        return fn
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = self.file_name_from_row(row)
        fn = os.path.join(self.point_cloud_root, fn)

        try: 
            
            assert os.path.exists(fn), f"Point cloud file {fn} does not exist"

            # Load the x, y, z and intensity coordinates of the point cloud
            # Read LAS for given plot ID
            laz_file = laspy.read(fn)
        
            # Cast x, y and z to float32 (to save memory and speed up processing)
            x = np.array(laz_file.x).astype(np.float32)
            y = np.array(laz_file.y).astype(np.float32)
            z = np.array(laz_file.z).astype(np.float32)
            intensity = laz_file.intensity
            laz_class_ids = laz_file.classification

            # Basic data points, x, y, z, intensity and laz class id (not to be confused with the class id in the dataset)
            points = [x, y, z, intensity, laz_class_ids]

            # Convert to N x C format e.g. 1000 X 5 where 5 is x, y, z, intensity and laz class id
            points = np.vstack(points).transpose()
        
            # Remove the points which belong to class 7 (noise) or 18 (noise)
            points = points[(points[:, 4] != 7) & (points[:, 4] != 18)]
            # Remove the class id column
            xyzi = points[:, :4]

            # Remove points which for whatever reason have a negative x, y or z value
            xyzi = xyzi[xyzi[:, 0] > 0]
            xyzi = xyzi[xyzi[:, 1] > 0]
            xyzi = xyzi[xyzi[:, 2] > 0]

            # Identify center of the point cloud and apply a circular mask using Pythagoras theorem
            center_x, center_y = (np.max(xyzi[:, 0]) + np.min(xyzi[:, 0])) / 2, (np.max(xyzi[:, 1]) + np.min(xyzi[:, 1])) / 2
            xyzi = self.apply_circle_mask(xyzi, self.radius, center_x, center_y)

            # Select a random subset of points given the self.num_points
            if self.num_points is not None and xyzi.shape[0] >= self.num_points:
                idx = np.random.choice(xyzi.shape[0], self.num_points, replace=False)
                xyzi = xyzi[idx, :]

            # Fill extra points into the points cloud if the number of points is less than the required number of points
            if self.num_points is not None and xyzi.shape[0] < self.num_points:
                n_points_to_fill = self.num_points - xyzi.shape[0]
                idx = np.random.choice(xyzi.shape[0], n_points_to_fill, replace=True)
                xyzi = np.concatenate([xyzi, xyzi[idx, :]], axis=0)

            data = {
                'pos': xyzi[:,:3],
                'y': row["class_id"]
            }

            # Apply the transform to the data 
            # train: [PointsToTensor, PointCloudScaling, PointCloudXYZAlign, PointCloudRotation, PointCloudJitter]
            # val: [PointsToTensor, PointCloudXYZAlign]
            if self.transform is not None:
                data = self.transform(data)

            # Append the gravity dimension to the data (height of the point cloud)
            point_heights = xyzi[:, self.gravity_dim:self.gravity_dim+1] - xyzi[:, self.gravity_dim:self.gravity_dim+1].min()
            point_heights_tensor = torch.from_numpy(point_heights) # 8192 x 1

            # Expand the dimension of the intensity from 8192 (1D) to 8192 x 1 (2D) and convert to tensor
            # intensity_tensor = torch.from_numpy(xyzi[:,3][:, np.newaxis])
            
            # Concatenate x, y, z, intensity and laz class id to the data['x']
            data['x'] = torch.cat((data['pos'], point_heights_tensor), dim=1)

            # Assert the data['x'] has the correct shape N x C=5
            assert data['x'].shape[1] == 4, f"Data['x'] has shape {data['x'].shape} instead of N x C=4"

            return fn, data
        except Exception as e:
            print(f"Error: {e} in file {fn}")
            return self.__getitem__(idx + 1)