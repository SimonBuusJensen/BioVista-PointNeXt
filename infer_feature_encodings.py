import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from openpoints.models import build_model_from_cfg
from openpoints.utils import dist_utils, EasyConfig, cal_model_parm_nums, load_checkpoint, set_random_seed, ConfusionMatrix
from openpoints.dataset import build_dataloader_from_cfg
from train_classifier import str2bool


if __name__ == "__main__":

    # Parse arguments
    argparse = argparse.ArgumentParser(description="Generate feature encodings for 3D Point clouds using a trained Point Vector Model")
    argparse.add_argument('--cfg', type=str, help='config file', 
                          default="/workspace/src/cfgs/biovista/pointvector-s.yaml")
                        #   default="cfgs/biovista/pointvector-s.yaml")
    argparse.add_argument("--dataset_csv", type=str, help="Path to an image, a directory of images or a csv file with image paths.",
                          default="/workspace/datasets/samples.csv")
                        #   default="/home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/samples.csv")
                        #   default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/samples.csv")
    argparse.add_argument("--model_weights", type=str, help="Path to the model weights file.",
                        #   default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        #   default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        #   default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                          default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
    argparse.add_argument("--batch_size", type=int, help="Batch size for the dataloader.", default=1)
    argparse.add_argument("--num_workers", type=int, help="Number of workers for the dataloader.", default=2)
    argparse.add_argument("--channels", type=str, help="Channels to use, x, y, z, h (height) and/or i (intensity)", default="xyzh")
    argparse.add_argument("--save_dir", type=str, help="Directory to save the feature encodings.", default=None)
    argparse.add_argument("--num_points", type=int, help="Number of points to sample from the point cloud.", default=16384)
    argparse.add_argument("--qb_radius", type=float, help="Query ball radius", default=0.65)
    argparse.add_argument("--qb_radius_scaling", type=float, help="Radius scaling factor", default=1.5)
    argparse.add_argument("--dataset_split", type=str, help="Dataset split to use for inference.", default="train")
    argparse.add_argument("--with_normalize_gravity_dim", type=str2bool, help="Whether to normalize the gravity dimension", default=False)
    argparse.add_argument("--with_normalize_intensity", type=str2bool, help="Whether to normalize the intensity", default=False)
    argparse.add_argument("--normalize_intensity_scale", type=float, help="Scale to factor to the normalization of the intensity", default=1.0)
    argparse.add_argument("--with_point_cloud_scaling", type=str2bool, help="Whether to use point cloud scaling data augmentation", default=False)
    argparse.add_argument("--with_point_cloud_rotations", type=str2bool, help="Whether to use point cloud rotation data augmentation", default=False)
    argparse.add_argument("--with_point_cloud_jitter", type=str2bool, help="Whether to use point cloud jitter data augmentation", default=False)
    argparse.add_argument("--seed", type=int, help="Random seed", default=1284)
    argparse.add_argument("--is_test_performance", type=str2bool, help="Whether to evaluate the model performance on the dataset set", default=False)

    # Parse the arguments
    args, opts = argparse.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    # Set the seed
    if args.seed is not None:
        cfg.seed = args.seed
    else:
        cfg.seed = np.random.randint(1, 10000)

    # Parse the csv and validate it
    dataset_csv = args.dataset_csv    
    assert os.path.exists(dataset_csv), f"Dataset csv file {dataset_csv} does not exist."
    print(f"Reading point cloud paths from {dataset_csv}...")
    cfg.dataset.common.data_root = args.dataset_csv
    df = pd.read_csv(dataset_csv)
    assert len(df) > 0, "The csv file is empty."

    # Set the point cloud file format
    cfg.dataset.common.format = "npz"

    # Set the batch size and number of workers
    cfg.dataloader.num_workers = args.num_workers # TODO check if this is correct
    assert args.batch_size > 0, "Batch size must be greater than 0"
    # Set the number of points in the point cloud and the channels

    # Set the number of points in the point cloud and the channels
    cfg.dataset.common.num_points = args.num_points
    cfg.model.encoder_args.in_channels = len(args.channels)
    cfg.dataset.common.channels = args.channels
    assert args.channels in ["xyz", "xyzi", "xyzh", "xyzhi", "xyzih"], "Channels must be one of xyz, xyzi, xyzh, xyzhi, xyzih"

    # Intensity normalization
    cfg.dataset.common.normalize_intensity = args.with_normalize_intensity
    cfg.dataset.common.normalize_intensity_scale = args.normalize_intensity_scale
    
    # Set the Query ball parameters
    cfg.model.encoder_args.radius = args.qb_radius
    cfg.model.encoder_args.radius_scaling = args.qb_radius_scaling

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

    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # Setup seed and device
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True

    # Set the model weights
    model_weights = args.model_weights
    assert os.path.exists(model_weights), f"Model weights file {model_weights} does not exist."

    # Set the save directory
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(model_weights)), "pointvector_encodings")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the dataset split
    dataset_split = args.dataset_split
    assert dataset_split in ["train", "val", "test"], "dataset_split must be one of ['train', 'test']"
    assert "dataset_split" in df.columns, "The csv file must contain a column 'dataset_split'."

    # Get rows with dataset_split == "test"
    df = df[df["dataset_split"] == dataset_split]
    print(f"Found {len(df)} image paths in the csv file.")

    # Set the dataset and data loader
    if dataset_split == "train":
        cfg.dataset.train.num_points = args.num_points
    elif dataset_split == "val":
        cfg.dataset.val.num_points = args.num_points
    elif dataset_split == "test":
        cfg.dataset.test.num_points = args.num_points
        cfg.dataset.test.seed = cfg.seed
    else:
        raise ValueError(f"dataset_split {dataset_split} is not supported.")
    
    data_loader = build_dataloader_from_cfg(args.batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split=dataset_split
                                           )
    # Filter away the samples which are already processed
    if not args.is_test_performance:
        df = data_loader.dataset.df.copy()
        existing_files = list(os.listdir(save_dir))
        missing_files = []
        for idx, row in df.iterrows():
            point_cloud_file_name = data_loader.dataset.file_name_from_row(row)
            point_cloud_file_name = point_cloud_file_name.replace(".npz", ".npy")
            if point_cloud_file_name not in existing_files:
                missing_files.append(row["id"])
        
        print(f"Found {len(missing_files)} missing files in the dataset.")
        df = df[df["id"].isin(missing_files)]
        
        data_loader.dataset.df = df

    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # test mode
    print("Loading model weights from: " + model_weights + "...")
    epoch, best_val = load_checkpoint(model, pretrained_path=model_weights)
    # Set model to gpu if available
 
    print(f"Number of samples in the {dataset_split} set: ", len(data_loader.dataset))
    model.eval()
    torch.backends.cudnn.enabled = True
    with torch.set_grad_enabled(False):
        
        if args.is_test_performance:
            pred_list = []
            conf_list = []
            label_list = []
            img_path_list = []
            confusion_matrix = ConfusionMatrix(num_classes=cfg.num_classes)

        pbar = tqdm(enumerate(data_loader), total=data_loader.__len__())
        for idx, (fns, data) in pbar:

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            target = data['y']
            points = data['x']
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

            features = model.encoder.forward_cls_feat(data)

            # Save the encodings
            for fn, feature in zip(fns, features):
                point_cloud_file_name = os.path.basename(fn)
                save_path = os.path.join(save_dir, point_cloud_file_name.replace(".npz", ".npy"))
                np.save(save_path, feature.cpu().numpy())
            
            if args.is_test_performance:
                logits = model(data)
                confusion_matrix.update(logits.argmax(dim=1), target)
                
                # Save the predictions and labels
                pred_list.extend(logits.argmax(dim=1).cpu().numpy())

                confidences = torch.nn.functional.softmax(logits, dim=1)
                confidences = torch.max(confidences, 1)[0]

                conf_list.extend(confidences.cpu().numpy())
                label_list.extend(target.cpu().numpy())
                img_path_list.extend(fns)
    
    if args.is_test_performance:
        # After looping through the dataset, compute overall metrics and save results
        tp, count = confusion_matrix.tp, confusion_matrix.count
        test_macc, test_oa, _ = confusion_matrix.cal_acc(tp, count)
        pred_label_fp = os.path.join(os.path.dirname(save_dir), "test_prediction_labels.csv")
        with open(pred_label_fp, "w") as f:
            f.write("image_path,prediction,label,correct,confidence\n")
            for img_path, pred, label, conf in zip(img_path_list, pred_list, label_list, conf_list):
                f.write(f"{os.path.basename(img_path)},{pred},{label},{int(pred==label)},{round(conf*100,0)}\n")
            # Write overall metrics
            low_total = confusion_matrix.actual[0].item()
            low_correct = confusion_matrix.tp[0].item() 
            low_acc = (low_correct/low_total)*100 if low_total>0 else 0
            f.write(f"Low bio correct,{low_correct},{low_total},{low_acc}\n")
            high_total = confusion_matrix.actual[1].item()
            high_correct = confusion_matrix.tp[1].item()
            high_acc = (high_correct/high_total)*100 if high_total>0 else 0
            f.write(f"High bio correct,{high_correct},{high_total},{high_acc}\n")
            f.write(f"Overall test accuracy,{confusion_matrix.tp.sum().item()},{confusion_matrix.actual.sum().item()},{test_oa}\n")
            f.write(f"Mean test accuracy,{test_macc}\n")