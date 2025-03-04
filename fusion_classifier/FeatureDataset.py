
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def orthophoto_file_name_from_csv_dataset_row_2(row, meters=30, is_nir=False):
        sample_id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        class_name = row["class_name"].replace(" ", "_").lower()
        year = row["year"]
        fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{sample_id}_{meters}m{'_nir' if is_nir else ''}.png"
        return fn

def laz_file_name_from_csv_dataset_row_2(row, format="npz"):
        id = row["sample_id"]
        ogc_fid = row["ogc_fid"]
        year = row["year"]
        class_name = row["class_name"].replace(" ", "_").lower()
        if format == 'npz':
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.npz"
        else:
            fn = f"{class_name}_{year}_ogc_fid_{ogc_fid}_{id}_30m.laz"
        return fn


class FeatureDataset(Dataset):
    def __init__(self, csv_file, feature_dir_2d, feature_dir_3d, data_split):
        assert os.path.exists(
            csv_file), f"CSV file: {csv_file} does not exist."
        self.df = pd.read_csv(csv_file)

        self.feature_dir_2D = feature_dir_2d
        self.feature_dir_3D = feature_dir_3d
        assert os.path.exists(
            self.feature_dir_2D), f"2D features directory {self.feature_dir_2D} does not exist."
        assert os.path.exists(
            self.feature_dir_3D), f"3D features directory {self.feature_dir_3D} does not exist."

        assert "dataset_split" in self.df.columns, "The column dataset_split' is missing from the dataframe."
        assert data_split in ["train", "val", "test"], f"Invalid data split: {data_split}. Must be one of ['train', 'val', 'test']"
        self.df = self.df[self.df["dataset_split"] == data_split]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        feature_file_name_2D = orthophoto_file_name_from_csv_dataset_row_2(
            row, 30)
        feature_file_2D = os.path.join(
            self.feature_dir_2D, feature_file_name_2D.replace(".png", ".npy"))
        features_2D = np.load(feature_file_2D)
        if len(features_2D.shape) == 2:
            features_2D = features_2D[0]
        
        feature_file_name_3D = laz_file_name_from_csv_dataset_row_2(row)
        feature_file_3D = os.path.join(
            self.feature_dir_3D, feature_file_name_3D.replace(".npz", ".npy"))
        features_3D = np.load(feature_file_3D)
        if len(features_3D.shape) == 2:
            features_3D = features_3D[0]

        # Concat the 2D and 3D feature vectors
        X = np.concatenate((features_2D, features_3D))
        X = torch.tensor(X, dtype=torch.float32)

        # Get label
        y = row["class_id"]
        y_one_hot = np.zeros(2)
        y_one_hot[y] = 1
        y_one_hot = torch.tensor(y_one_hot, dtype=torch.float32)

        return feature_file_name_2D, X, y_one_hot