from torch.utils.data import Dataset
from pathlib import Path
import os 
import pandas as pd
import numpy as np
import laspy
import matplotlib.pyplot as plt 
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
                 format='npz',
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
            points = data_npz['points'] # 
            return points
        
        elif self.format == 'laz':
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
            raise ValueError(f"Unknown format {self.format}")
    
    def getitem_by_class_id_year_ogc_fid_and_sample_id(self, class_id, year, ogc_fid, sample_id):
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if row['class_id'] == class_id and row['year'] == year and row['ogc_fid'] == ogc_fid and row['sample_id'] == sample_id:
                idx = i
                break
           
        return self.__getitem__(idx)
    
    def create_colormap(self, num_colors=400):

        # Create a colormap using matplotlib
        cmap = plt.get_cmap('jet')

        # Generate colors for each index
        colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

        # Convert to 8-bit RGB format
        colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

        return colors_rgb
    
    def points_2_ply(self, points, fn, color_mode='rgb', n_colors=256):
        # Save the points as a ply file (x, y, z, r, g, b)

        num_points = points.shape[0]

        # Create a directory for the file if it doesn't exist
        Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)

        if color_mode == 'intensity':
            max_intensity = np.max(points[:, 6])
            min_intensity = np.min(points[:, 6])
            print(f"Max intensity: {max_intensity}, Min intensity: {min_intensity}")
            # Normalize the intensity values to 0-n_colors
            normalized_intensities = (points[:, 6] - min_intensity) / (max_intensity - min_intensity) * (n_colors - 1)
            cmap = self.create_colormap(n_colors)
        
        if color_mode == 'height':
            max_height = np.max(points[:, 2])
            min_height = np.min(points[:, 2])
            print(f"Max height: {max_height}, Min height: {min_height}")
            # Normalize the height values to 0-n_colors
            normalized_heights = (points[:, 2] - min_height) / (max_height - min_height) * (n_colors - 1)
            cmap = self.create_colormap(n_colors)

        with open(fn, 'w') as file:
        # Writing the PLY header
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {num_points}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")   # 8-bit color
            file.write("property uchar green\n") # 8-bit color
            file.write("property uchar blue\n")  # 8-bit color
            file.write("end_header\n")

            # Writing the point data
            for i in range(num_points):
                x, y, z, r, g, b, *_ = points[i]  # Use * to ignore extra columns

                if color_mode == 'rgb':
                    file.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
                elif color_mode == 'intensity':
                    color = cmap[int(normalized_intensities[i])]
                    file.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")
                elif color_mode == 'height':
                    color = cmap[int(normalized_heights[i])]
                    file.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")

            file.close()
        
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = self.file_name_from_row(row)
        fn = os.path.join(self.point_cloud_root, fn)

        # Save dir for the ply files
        save_dir = "/home/simon/data/BioVista/Forest-Biodiversity-Potential/figures/point_cloud_processing_pipeline/"
        color_mode = 'intensity'
        try: 
            
            assert os.path.exists(fn), f"Point cloud file {fn} does not exist"

            points = self.load_point_cloud(fn) # x, y, z, r, g, b, intensity, class_id (laz class id)

            # Save the points as a ply file (x, y, z, r, g, b)
            fn_after_load = os.path.join(save_dir, "1_" + os.path.basename(fn).replace(f".{self.format}", f"_{color_mode}.ply"))
            print("---------------------------------")
            print("Step 1: Load the point cloud")
            self.points_2_ply(points, fn_after_load, color_mode=color_mode)
            
            # Remove the points which belong to class 7 (noise) or 18 (noise)
            print("---------------------------------")
            print("Step 2: Remove noise points")
            print(f"Before removing noise points: {points.shape}")
            points = points[(points[:, -1] != 7) & (points[:, -1] != 18)]
            print(f"After removing noise points: {points.shape}")
            
            # Remove points which for whatever reason have a negative x, y or z value
            print("---------------------------------")
            print("Step 3: Remove points with negative x, y or z value")
            print(f"Before removing negative x, y or z values: {points.shape}")
            points = points[points[:, 0] > 0]
            points = points[points[:, 1] > 0]
            points = points[points[:, 2] > 0]
            print(f"After removing negative x, y or z values: {points.shape}")

            # Identify center of the point cloud and apply a circular mask using Pythagoras theorem
            print("---------------------------------")
            print("Step 4: Apply circular mask")
            center_x, center_y = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2, (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
            points = self.apply_circle_mask(points, self.radius, center_x, center_y)
            fn_after_circle_mask = os.path.join(save_dir, "2_" + os.path.basename(fn).replace(f".{self.format}", f"_{color_mode}_circle_mask.ply"))
            self.points_2_ply(points, fn_after_circle_mask, color_mode=color_mode)

            # Select a random subset of points given the self.num_points
            if self.num_points is not None and points.shape[0] >= self.num_points:
                print("---------------------------------")
                print("Step 5: Select a random subset of points")
                print(f"Number of points before random subset: {points.shape}")
                idx = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[idx, :]
                fn_after_random_subset = os.path.join(save_dir, "3_" + os.path.basename(fn).replace(f".{self.format}", f"_{color_mode}_random_subset.ply"))
                print("Number of points after random subset: ", points.shape)
                self.points_2_ply(points, fn_after_random_subset, color_mode=color_mode)

            # Fill extra points into the points cloud if the number of points is less than the required number of points
            if self.num_points is not None and points.shape[0] < self.num_points:
                n_points_to_fill = self.num_points - points.shape[0]
                idx = np.random.choice(points.shape[0], n_points_to_fill, replace=True)
                points = np.concatenate([points, points[idx, :]], axis=0)

            data = {
                'pos': points[:,:3],
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
                point_heights = points[:, self.gravity_dim:self.gravity_dim+1] - points[:, self.gravity_dim:self.gravity_dim+1].min()
                point_heights_tensor = torch.from_numpy(point_heights) # 8192 x 1
                data['x'] = torch.cat((data['x'], point_heights_tensor), dim=1)
            
            # Append the intensity dimension to the data
            if "i" in self.channels:
                intensity_tensor = torch.from_numpy(points[:, 6][:, np.newaxis])
                data['x'] = torch.cat((data['x'], intensity_tensor), dim=1)

            # Assert the data['x'] has the correct shape N x C=5
            assert data['x'].shape[1] == self.n_channels, f"Data['x'] has shape {data['x'].shape} instead of N x C={self.n_channels}"

            return fn, data
        except Exception as e:
            print(f"Error: {e} in file {fn}")
            return self.__getitem__(idx + 1)
    
    def __len__(self):
        return len(self.df)

if __name__ == "__main__":

    color_drop = 0.2
    angle = [0, 0, 1]
    jitter_sigma = 0.005
    jitter_clip= 0.02
    transforms = Compose([
        PointsToTensor(),
        PointCloudScaling(scale=[0.9, 1.1]),
        PointCloudXYZAlign(),
        PointCloudRotation(),
        PointCloudJitter(jitter_clip=jitter_clip, jitter_sigma=jitter_sigma)
    ])

    dataset = BioVista(
        data_root="/home/simon/data/BioVista/Forest-Biodiversity-Potential/samples.csv", 
        split='train', 
        transform=transforms,
        format='npz',
        num_points=8192
    )
    print(len(dataset))
    
    dataset.getitem_by_class_id_year_ogc_fid_and_sample_id(1, 2021, 18, 120)
