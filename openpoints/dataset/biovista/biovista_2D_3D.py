from torch.utils.data import Dataset
# from ..build import DATASETS
import os
import logging
import pandas as pd
import numpy as np
import laspy
import torch
from PIL import Image

import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '../../..')
sys.path.insert(0, parent_dir)
from resnet_nir.datatransforms import set_val_data_transforms, set_train_data_transforms



# @DATASETS.register_module()
class BioVista2D3D(Dataset):

    num_classes = 2
    classes = ['low_bio', 'high_bio']
    gravity_dim = 2
    CM_PER_PX = 12.5 # 12.5 cm per pixel in the orthophoto

    def __init__(self,
                 data_root='/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv',
                 split='train',
                 num_points=16384,
                 transform=None,
                 format='npz',
                 seed=None,
                 ):

        # Assert the data_root is a csv file
        assert data_root.endswith('.csv')
        csv_file = data_root
        self.df = pd.read_csv(csv_file)
        self.data_root = os.path.dirname(csv_file)

        # General Settings
        self.test_plot_radius_meters = 15
        self.split = split
        # assert split is either train, val or test
        assert split in ['train', 'val', 'test']
        self.df = self.df[self.df['dataset_split'] == split]
        # Shuffle the dataframe
        if split == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        # Make sure test is deterministic with seed
        if split == "test":
            if seed is None:
                self.seed = 42
            else:
                self.seed = seed

        """ 
        ALS Settings
        """
        self.num_points = num_points
        self.als_channels = "xyzh"
        self.n_als_channels = len(self.als_channels)
        self.format = format
        self.transform = transform
        assert self.format in ['npz', 'laz']
        if self.format == 'laz':
            self.point_cloud_root = os.path.join(self.data_root, f"ALS_point_clouds")
        else:
            self.point_cloud_root = os.path.join(self.data_root, f"ALS_point_clouds_npz")
        assert os.path.exists(self.point_cloud_root), f"Point cloud root {self.point_cloud_root} does not exist"
        assert len(os.listdir(self.point_cloud_root)) > 0, f"Point cloud root {self.point_cloud_root} is empty"

        """ 
        Orthophoto Settings
        """
        self.rgb_image_root = os.path.join(self.data_root, "RGB_orthophotos")
        self.nir_image_root = os.path.join(self.data_root, "NIR-RG_orthophotos")
        self.orthophoto_channels = "NGB"
        self.test_plot_diameter_meters = self.test_plot_radius_meters * 2
        self.test_plot_diameter_pxs = int(self.test_plot_diameter_meters * 100 / self.CM_PER_PX)
        if self.split == "val" or self.split == "test":
            self.orthophoto_transform = set_val_data_transforms(self.test_plot_diameter_pxs)
        else:
            self.orthophoto_transform = set_train_data_transforms(self.test_plot_diameter_pxs)
    
    def apply_circlular_mask_on_orthophoto(self, img_array, diameter_px):
        """
        Apply a circular alpha mask to a numpy array.
        
        Args:
            img_array (np.ndarray): Image array with shape (height, width, channels)
            diameter_px (int): Diameter of the circular mask in pixels
        
        Returns:
            np.ndarray: Image array with circular mask applied
        """
        result = img_array.copy()
        
        # Create circular mask using numpy broadcasting
        y, x = np.ogrid[:diameter_px, :diameter_px]
        center = diameter_px // 2
        mask = ((x - center)**2 + (y - center)**2 <= (center)**2).astype(img_array.dtype)
        
        # Apply mask to all channels
        result[:diameter_px, :diameter_px] *= mask[..., np.newaxis]
        return result

    def apply_circlular_mask_on_point_cloud(self, point_cloud_array, radius, center_x, center_y):
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
        return f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}"

    def als_file_name_from_row(self, row):
        id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        year = row["year"]
        class_name = row["class_name"].replace(" ", "_").lower()
        if self.format == 'npz':
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.npz"
        else:
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.laz"
        return fn
    
    def orthophoto_file_name_from_row(self, row, is_nir=False):
        sample_id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        class_name = row["class_name"].replace(" ", "_").lower()
        year = row["year"]
        fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{sample_id}_30m{'_nir' if is_nir else ''}.png"
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
    
    def filter_points(self, points):
        # Remove the points which belong to class 7 (noise) or 18 (noise)
        points = points[(points[:, 4] != 7) & (points[:, 4] != 18)]

        # Remove points which for whatever reason have a negative z values (below the ground)
        points = points[points[:, 2] > 0]
        return points
    
    def append_height_feature(self, data, points):
        point_heights = points[:, self.gravity_dim:self.gravity_dim + 1] - points[:, self.gravity_dim:self.gravity_dim+1].min()
        point_heights_tensor = torch.from_numpy(point_heights) 
        data['x'] = torch.cat((data['x'], point_heights_tensor), dim=1)
        return data
     
    def get_points_by_row(self, row):
        fn = self.als_file_name_from_row(row)
        fn = os.path.join(self.point_cloud_root, fn)
        assert os.path.exists(fn), f"Point cloud file {fn} does not exist"

        # Load the point cloud and filter the points
        points = self.load_point_cloud(fn)
        points = self.filter_points(points) # Filter by class_id and z values

        # Identify center of the point cloud and apply a circular mask using Pythagoras theorem
        center_x, center_y = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2, (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
        points = self.apply_circlular_mask_on_point_cloud(points, self.test_plot_radius_meters, center_x, center_y)

        if self.split == 'test':
            np.random.seed(self.seed)
        
        if points.shape[0] >= self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices, :]
        else:
            indices = np.random.choice(points.shape[0], self.num_points - points.shape[0], replace=True)
            points = np.concatenate([points, points[indices, :]], axis=0)
        return points
    

    def get_orthophoto_by_row(self, row):
        
        # Check if the RGB orthophoto exists
        rgb_fn = self.orthophoto_file_name_from_row(row, is_nir=False)
        rgb_img_path = os.path.join(self.rgb_image_root, rgb_fn)
        assert os.path.exists(rgb_img_path), f"Image file: {rgb_img_path} does not exist."

        # Check if the NIR orthophoto exists
        nir_fn = self.orthophoto_file_name_from_row(row, is_nir=True)
        nir_img_path = os.path.join(self.nir_image_root, nir_fn)
        assert os.path.exists(nir_img_path), f"Image file: {nir_img_path} does not exist."

        # Load the image and label
        rgb_img = Image.open(rgb_img_path).convert("RGB")
        rgb_array = np.array(rgb_img)
        nir_img = Image.open(nir_img_path).convert("RGB")
        nir_array = np.array(nir_img)

        # Create an empty image array and fill it with the NIR, Green and Blue channels
        image_array = np.zeros((self.test_plot_diameter_pxs, self.test_plot_diameter_pxs, len(self.orthophoto_channels)), dtype=np.float32)
        if self.orthophoto_channels == "NGB":
            image_array[:, :, 0] = nir_array[:, :, 0]
            image_array[:, :, 1] = rgb_array[:, :, 1]
            image_array[:, :, 2] = rgb_array[:, :, 2]
        
        # Convert to uint8
        image_array = image_array.astype(np.uint8)

        # Apply transformations
        if self.orthophoto_transform:
            # Convert to a PIL image and apply the transformation
            img = Image.fromarray(image_array)
            img = self.orthophoto_transform(img)
            # Convert to back to a numpy array after the transformation
            image_array = np.array(img) 
        
        # Apply circular mask
        image_array = self.apply_circlular_mask_on_orthophoto(img_array=image_array, diameter_px=self.test_plot_diameter_pxs)
        
        # Convert to Tensor and Convert from HWC to CHW
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor


    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        sample_name = self.file_name_from_row(row)
        
        """
        Prepare the ALS data
        """
        points = self.get_points_by_row(row)
        
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
        if "h" in self.als_channels:
            self.append_height_feature(data, points)

        """ 
        Prepare the orthophoto data
        """
        orthophoto = self.get_orthophoto_by_row(row)

        data['img'] = orthophoto

        return sample_name, data
        

    def __len__(self):
        return len(self.df)


# Test the BioVista2D3D dataset
if __name__ == "__main__":

    from torchvision.transforms import Compose
    from openpoints.transforms import PointsToTensor, PointCloudXYZAlign
    transform = Compose([PointsToTensor(), PointCloudXYZAlign()])

    dataset = BioVista2D3D(data_root='/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv', split='train', transform=transform)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])