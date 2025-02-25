import argparse

import numpy as np

from openpoints.utils import EasyConfig, set_random_seed
from train_classifier import str2bool




if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training a BioVista dataset classifier consisting of a PointVector and a ResNet Model which are then fused together with a MLP for fusion')
    parser.add_argument('--pointvector_cfg', type=str, help='config file', 
                        # default="cfgs/biovista_2D_3D/pointvector-s.yaml")
                        default="/workspace/src/cfgs/biovista_2D_3D/pointvector-s.yaml")
    # PointVector Settings


    # ResNet Settings:

    # MLP Settings


    # General Settings:
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument("--wandb", type=str2bool, help="Whether to log to weights and biases", default=True)
    parser.add_argument("--project_name", type=str, help="Weights and biases project name", default="BioVista-MLP-Fusion-2D-3D-Active-Learning-v1")

    
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.pointvector_cfg, recursive=True)
    cfg.update(opts)

    # Set the seed
    if args.seed is not None:
        cfg.seed = args.seed
    else:
        cfg.seed = np.random.randint(1, 10000)
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
