import argparse
import os

import numpy as np

from openpoints.utils import EasyConfig, set_random_seed
from train_classifier import str2bool
from fusion_classifier.MLPModel import MLPModel




if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training a BioVista dataset classifier consisting of a PointVector and a ResNet Model which are then fused together with a MLP for fusion')
    # Dataset settings
    parser.add_argument("--csv", type=str, help="Path to a csv file containing the name of the training and validation data files.",
                        default="/workspace/datasets/samples.csv")

    # PointVector Settings
    parser.add_argument('--pointvector_cfg', type=str, help='config file', 
                        # default="cfgs/biovista_2D_3D/pointvector-s.yaml")
                        default="/workspace/src/cfgs/biovista_2D_3D/pointvector-s.yaml")
    parser.add_argument("--features_dir_2d", type=str, help="Path to a directory containing the 2D features of the images.",
                        default="/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/resnet_encodings")


    # ResNet Settings:
    parser.add_argument("--features_dir_3d", type=str, help="Path to a directory containing the 3D features of the point clouds.",
                        default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/pointvector_encodings")

    # MLP Settings
    

    # General Settings:
    parser.add_argument("--is_active_weights", type=str2bool, help="Whether to freeze the weights of the PointVector and ResNet models", default=False)
    parser.add_argument("--mode", type=str, help="Mode (train or test)", default="train")
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument("--wandb", type=str2bool, help="Whether to log to weights and biases", default=True)
    parser.add_argument("--project_name", type=str, help="Weights and biases project name", default="BioVista-MLP-Fusion-2D-3D-Active-Weights-Test-Version")

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.pointvector_cfg, recursive=True)
    cfg.update(opts)

    csv_file = args.csv
    assert os.path.exists(csv_file), "The csv file does not exist."
    assert csv_file.endswith(".csv"), "The csv file must have a .csv extension."

    # Set the seed
    if args.seed is not None:
        assert 1 <= args.seed <= 10000, "The seed must be between 1 and 10000."
        assert isinstance(args.seed, int), "The seed must be an integer."
        cfg.seed = args.seed
    else:
        cfg.seed = np.random.randint(1, 10000)

    set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    # Set the mode
    cfg.mode = args.mode
    assert cfg.mode in ["train", "test"]

    # Set with or without active learning (is_active_weights)
    cfg.is_active_weights = args.is_active_weights

    # Assert the features_dir_2d and features_dir_3d exists and are none empty directories in is_active_weights is False
    if not cfg.is_active_weights:
        assert args.features_dir_2d is not None and args.features_dir_3d is not None
        assert os.path.exists(args.features_dir_2d), "The 2D features directory does not exist."
        assert os.path.exists(args.features_dir_3d), "The 3D features directory does not exist."
        assert len(os.listdir(args.features_dir_2d)) == 44378, "The 2D features does not contain 44378 files."
        assert len(os.listdir(args.features_dir_3d)) == 44378, "The 3D features does not contain 44378 files."

    # Init the MLP model
    MLPModel()