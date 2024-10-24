import os
import argparse
import yaml
import wandb
import numpy as np
from datetime import datetime
from torch import multiprocessing as mp
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, help='config file', default="/workspace/src/cfgs/biovista/pointvector-s.yaml")
    parser.add_argument('--dataset_csv', type=str, help='dataset csv file', default="/workspace/datasets/high_and_low_HNV-forest-proxy-train-val-dataset/high_and_low_HNV-forest-proxy-train-val-polygon-dataset_30_m_circles_dataset_without_empty_clouds.csv")
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument("--num_points", type=int, help="Number of points in the point cloud", default=8192)
    parser.add_argument("--qb_radius", type=float, help="Query ball radius", default=0.7)
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=10)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=2)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--wandb", type=bool, help="Whether to log to weights and biases", default=True)
    parser.add_argument("--project_name", type=str, help="Weights and biases project name", default="BioVista-3D-ALS")

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.num_points is not None:
        cfg.dataset.train.num_points = args.num_points
        cfg.dataset.val.num_points = args.num_points
        cfg.dataset.test.num_points = args.num_points
        cfg.num_points = args.num_points

    if args.qb_radius is not None:
        cfg.model.encoder_args.radius = args.qb_radius

    if args.batch_size_train is not None:
        cfg.batch_size = args.batch_size_train
        cfg.val_batch_size = cfg.batch_size

    if args.lr is not None:
        cfg.lr = args.lr

    # Set the dataset csv file
    cfg.dataset.common.data_root = args.dataset_csv
    cfg.root_dir = os.path.join(os.path.dirname(args.dataset_csv), "experiments")

    # Parse the model name from the cfg file
    model_name = os.path.basename(args.cfg).split('.')[0]
    date_now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{date_now_str}_{args.project_name}_{model_name}_batch-sz_{cfg.batch_size}_{cfg.num_points}_lr_{cfg.lr}_qb-radius_{cfg.model.encoder_args.radius}"
    

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

    if cfg.mode == 'pretrain':
        main = pretrain
    else:
        main = train

    main(0, cfg, profile=args.profile)
