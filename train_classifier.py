import os
import argparse
import yaml
import wandb
import numpy as np
from datetime import datetime
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, generate_exp_directory, resume_exp_directory

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, help='config file',
                        # default="/workspace/src/cfgs/biovista/pointvector-s.yaml")
                        default="cfgs/biovista/pointvector-s.yaml")
    parser.add_argument('--dataset_csv', type=str, help='dataset csv file', 
                        # default="/workspace/datasets/samples.csv") 
                        default="/home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/samples.csv")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=20)
    parser.add_argument("--pcl_file_format", type=str, help="Point cloud file format (npz | laz)", default="npz")
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2)
    parser.add_argument("--num_workers", type=int, help="The number of threads for the dataloader", default=0)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--num_points", type=int, help="Number of points in the point cloud", default=8192)
    parser.add_argument("--channels", type=str, help="Channels to use, x, y, z, h (height) and/or i (intensity)", default="xyzhi")
    parser.add_argument("--qb_radius", type=float, help="Query ball radius", default=0.65)
    parser.add_argument("--qb_radius_scaling", type=float, help="Radius scaling factor", default=1.5)
    parser.add_argument("--with_class_weights", type=str2bool, help="Whether to use class weights", default=False)
    parser.add_argument("--with_normalize_gravity_dim", type=str2bool, help="Whether to normalize the gravity dimension", default=False)
    parser.add_argument("--with_normalize_intensity", type=str2bool, help="Whether to normalize the intensity", default=False)
    parser.add_argument("--normalize_intensity_scale", type=float, help="Scale to factor to the normalization of the intensity", default=1.0)
    parser.add_argument("--with_point_cloud_scaling", type=str2bool, help="Whether to use point cloud scaling data augmentation", default=True)
    parser.add_argument("--with_point_cloud_rotations", type=str2bool, help="Whether to use point cloud rotation data augmentation", default=True)
    parser.add_argument("--with_point_cloud_jitter", type=str2bool, help="Whether to use point cloud jitter data augmentation", default=True)
    parser.add_argument("--wandb", type=str2bool, help="Whether to log to weights and biases", default=True)
    parser.add_argument("--project_name", type=str, help="Weights and biases project name", default="BioVista-Intensity-Experiments_v1")

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    
    # Set the seed
    cfg.seed = np.random.randint(1, 10000)
    
    # Set the dataset csv file
    cfg.dataset.common.data_root = args.dataset_csv
    assert os.path.exists(cfg.dataset.common.data_root), f"Dataset csv file {cfg.dataset.common.data_root} does not exist"
    cfg.root_dir = os.path.join(os.path.dirname(args.dataset_csv), "experiments")

    # Set the point cloud file format
    cfg.dataset.common.format = args.pcl_file_format

    # Set epochs, batch size and learning rate
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.dataloader.num_workers = args.num_workers # TODO check if this is correct
    assert args.batch_size > 0 and args.epochs > 0, "Batch size and epochs must be greater than 0"
    cfg.val_batch_size = args.batch_size
    cfg.lr = args.lr
    
    # Set the number of points in the point cloud and the channels
    cfg.num_points = args.num_points
    cfg.model.encoder_args.in_channels = len(args.channels)
    cfg.dataset.common.channels = args.channels
    cfg.dataset.common.normalize_intensity = args.with_normalize_intensity
    cfg.dataset.common.normalize_intensity_scale = args.normalize_intensity_scale
    assert args.channels in ["xyz", "xyzi", "xyzh", "xyzhi", "xyzih"], "Channels must be one of xyz, xyzi, xyzh, xyzhi, xyzih"

    # Set the Query ball parameters
    cfg.model.encoder_args.radius = args.qb_radius
    cfg.model.encoder_args.radius_scaling = args.qb_radius_scaling

    # With/without class weights
    cfg.cls_weighed_loss = args.with_class_weights
    
    # Data Augmentation list
    if args.with_point_cloud_scaling:
        cfg.datatransforms.train.append("PointCloudScaling")
    if args.with_point_cloud_rotations:
        cfg.datatransforms.train.append("PointCloudRotation")
    if args.with_point_cloud_jitter:
        cfg.datatransforms.train.append("PointCloudJitter")
    if args.with_normalize_gravity_dim:
        cfg.datatransforms.kwargs.normalize_gravity_dim = True
    else:
        cfg.datatransforms.kwargs.normalize_gravity_dim = False
    
    # Parse the model name from the cfg file
    model_name = os.path.basename(args.cfg).split('.')[0]
    date_now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{date_now_str}_{args.project_name}_{model_name}_channels_{args.channels}_npts_{args.num_points}_qb_r_{args.qb_radius}_qb_s_{args.qb_radius_scaling}"
    
    if args.wandb:
        cfg.wandb.use_wandb = True
        cfg.wandb.project = args.project_name
        wandb.init(project=cfg.wandb.project, name=experiment_name)
        wandb.config.update(args)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.opts = '-'.join(opt_list)

    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None), run_name=experiment_name)
        cfg.wandb.tags = tags
        
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    # Add the cfg and log file to the wandb
    if args.wandb:
        wandb.save(cfg.cfg_path)
        wandb.save(cfg.log_dir)

    train(0, cfg)
