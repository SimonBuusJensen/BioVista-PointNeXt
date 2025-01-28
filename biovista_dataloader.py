from torch.utils.data import Dataset
import os 
import pandas as pd
import numpy as np
import laspy
import torch
from torchvision.transforms import Compose
from openpoints.transforms import PointsToTensor, PointCloudScaling, PointCloudXYZAlign, PointCloudRotation, PointCloudJitter

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
                 format='npz'
                 ):
        
        # Assert the data_root is a csv file
        assert data_root.endswith('.csv')
        csv_file = data_root
        self.df = pd.read_csv(csv_file)
        
        self.num_points = num_points
        self.channels = channels
        self.n_channels = len(channels)

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
        mask = (point_cloud_array[:, 0] - center_x)**2 + (point_cloud_array[:, 1] - center_y)**2 <= radius**2 # Use Pythagoras theorem to find the points inside the circle
        point_cloud_array = point_cloud_array[mask]
        return point_cloud_array

    def __len__(self):
        return len(self.df)

    def file_name_from_row(self, row):
        id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        year = row["year"]
        class_name = row["class_name"].replace(" ", "_").lower()
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
            points = data_npz['points'] # 
            
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
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = self.file_name_from_row(row)
        fn = os.path.join(self.point_cloud_root, fn)

        try: 
            
            assert os.path.exists(fn), f"Point cloud file {fn} does not exist"

            points = self.load_point_cloud(fn)

            # Remove the points which belong to class 7 (noise) or 18 (noise)
            points = points[(points[:, 4] != 7) & (points[:, 4] != 18)]
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

            data['x'] = data['pos']
            
            # Append the gravity dimension to the data (height of the point cloud)
            if "h" in self.channels:
                point_heights = xyzi[:, self.gravity_dim:self.gravity_dim+1] - xyzi[:, self.gravity_dim:self.gravity_dim+1].min()
                point_heights_tensor = torch.from_numpy(point_heights) # 8192 x 1
                data['x'] = torch.cat((data['x'], point_heights_tensor), dim=1)
            
            # Append the intensity dimension to the data
            if "i" in self.channels:
                intensity_tensor = torch.from_numpy(xyzi[:,3][:, np.newaxis])
                data['x'] = torch.cat((data['x'], intensity_tensor), dim=1)

            # Assert the data['x'] has the correct shape N x C=5
            assert data['x'].shape[1] == self.n_channels, f"Data['x'] has shape {data['x'].shape} instead of N x C={self.n_channels}"

            return fn, data
        except Exception as e:
            print(f"Error: {e} in file {fn}")
            return self.__getitem__(idx + 1)
        

if __name__ == "__main__":

    transforms = Compose([
        PointsToTensor(),
        PointCloudScaling(),
        PointCloudXYZAlign(),
        PointCloudRotation(),
        PointCloudJitter()
    ])

    dataset = BioVista(
        data_root="/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv", 
        split='train', 
        transform=transforms,
        format='laz'
    )
    print(len(dataset))
    print(dataset[0])