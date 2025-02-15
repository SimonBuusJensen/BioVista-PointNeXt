import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, cal_model_parm_nums, load_checkpoint, set_random_seed, ConfusionMatrix
from openpoints.dataset import build_dataloader_from_cfg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def is_valid_source(source: str) -> bool:
    """
    Check if the source is either a .laz file, a directory of .laz files or a .csv file with valid paths.
    """
    try:
        assert os.path.exists(source), f"Source {source} does not exist."
        assert source.endswith(".laz") or source.endswith(".csv") or os.path.isdir(source), f"Source {source} must be a .laz file, a directory of .laz files or a .csv file."
        is_file = os.path.isfile(source)
        is_dir = not is_file and os.path.isdir(source)
        assert is_file != is_dir, f"Source {source} must be a .laz file, a directory of .laz files or a .csv file."
        if is_dir:
            "Check that .laz files exist in the directory."
            files = os.listdir(source)
            assert len(files) > 0, f"Directory {source} is empty."
            contains_laz_files = False
            for file in files:
                if file.endswith(".laz"):
                    contains_laz_files = True
                    break 
            assert contains_laz_files, f"Directory {source} does not contain .laz files."
        return True
    except AssertionError as e:
        print(e)
        return False


if __name__ == "__main__":

    # Parse arguments
    argparse = argparse.ArgumentParser(description="Generate feature encodings for 3D Point clouds using a trained Point Vector Model")
    argparse.add_argument('--cfg', type=str, help='config file', 
                        #   default="/workspace/src/cfgs/biovista/pointvector-s.yaml")
                          default="cfgs/biovista/pointvector-s.yaml")
    argparse.add_argument("--source", type=str, help="Path to an image, a directory of images or a csv file with image paths.",
                        #   default="/workspace/datasets/samples.csv")
                          default="/home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/samples.csv")
    argparse.add_argument("--model_weights", type=str, help="Path to the model weights file.",
                          default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-pointc-cloud-PointVector/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        #   default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-pointc-cloud-PointVector/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        #   default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-pointc-cloud-PointVector/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        #   default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-pointc-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
    argparse.add_argument("--save_dir", type=str, help="Directory to save the feature encodings.", default=None)
    argparse.add_argument("--batch_size", type=int, help="Batch size for the dataloader.", default=8)
    argparse.add_argument("--qb_radius", type=float, help="Query ball radius", default=0.65)
    argparse.add_argument("--qb_radius_scaling", type=float, help="Radius scaling factor", default=1.5)
    argparse.add_argument("--num_points", type=int, help="Number of points to sample from the point cloud.", default=16384)
    argparse.add_argument("--num_workers", type=int, help="Number of workers for the dataloader.", default=4)
    argparse.add_argument("--dataset_split", type=str, help="Dataset split to use for inference.", default="test")
    argparse.add_argument("--channels", type=str, help="Channels to use, x, y, z, h (height) and/or i (intensity)", default="xyzh")
    argparse.add_argument("--pcld_format", type=str, help="File format of the dataset.", default="npz")
    argparse.add_argument("--with_normalize_gravity_dim", type=str2bool, help="Whether to normalize the gravity dimension", default=False)
    argparse.add_argument("--is_test_performance", type=str2bool, help="Whether to evaluate the model performance on the dataset set", default=False)
    argparse.add_argument("--seed", type=int, help="Random seed", default=9447)

    args, opts = argparse.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(1, 10000)

    set_random_seed(seed, deterministic=cfg.deterministic)

    # Parse the source and validate it
    source = args.source    
    if not is_valid_source(source):
        raise ValueError(f"Invalid source {source}.")

    source_is_file = os.path.isfile(source)
    source_is_dir = not source_is_file

    # Assert pcld_format is either npz or laz
    assert args.pcld_format in ["npz", "laz"], f"Point cloud format {args.pcld_format} is not supported."
    cfg.dataset.common.format = args.pcld_format

    if source_is_dir:
        n_laz_files = len([f for f in os.listdir(source) if f.endswith(".laz")])
        print(f"Found {n_laz_files} .laz files in the directory {source}.")
    if source_is_file and source.endswith(".csv"):
        print(f"Reading image paths from {source}...")
        df = pd.read_csv(source)
        dataset_split = args.dataset_split
        assert dataset_split in ["train", "val", "test"], "dataset_split must be one of ['train', 'test']"
        assert "dataset_split" in df.columns, "The csv file must contain a column 'dataset_split'."

        # Get rows with dataset_split == "test"
        df = df[df["dataset_split"] == dataset_split]
        print("-------------------------------------------------------------------")
        print("WARNING! only using the test samples in the csv file for inference.")
        print("-------------------------------------------------------------------")
        print(f"Found {len(df)} image paths in the csv file.")
    
    cfg.dataset.common.data_root = source

    # Load the model given the config file cfg.yaml
    cfg.model.encoder_args.in_channels = len(args.channels)
    cfg.dataset.common.channels = args.channels
    model = build_model_from_cfg(cfg.model)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # Load the model weights
    model_weights = args.model_weights
    assert os.path.exists(model_weights), f"Model weights file {model_weights} does not exist."

    # test mode
    print("Loading model weights from: " + model_weights + "...")
    epoch, best_val = load_checkpoint(model, pretrained_path=model_weights)
    # Set model to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the save directory
    # Set the save_dir if not provided
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(model_weights)), "pointvector_encodings")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg.model.in_channels = cfg.model.encoder_args.in_channels
    print(f"Loaded model weights from epoch {epoch} with best validation accuracy {best_val}")

    # Setup the dataset and data loader
    batch_size = args.batch_size
    assert batch_size > 0, "Batch size must be greater than 0."
    num_workers = args.num_workers
    cfg.dataloader.num_workers = num_workers
    if num_workers == 0:
        print("Warning: num_workers is set to 0. This might slow down the training process.")

    # Set the number of points in the point cloud and the channels
    cfg.dataset.common.num_points = args.num_points

    # Set the Query ball parameters
    cfg.model.encoder_args.radius = args.qb_radius
    cfg.model.encoder_args.radius_scaling = args.qb_radius_scaling

    if args.with_normalize_gravity_dim:
        cfg.datatransforms.kwargs.normalize_gravity_dim = True
    else:
        cfg.datatransforms.kwargs.normalize_gravity_dim = False

    data_loader = build_dataloader_from_cfg(batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split=dataset_split,
                                           distributed=False
                                           )
    
    if args.is_test_performance:
        confusion_matrix = ConfusionMatrix(num_classes=cfg.num_classes)
        # Init containers to store the predictions, confidences, labels and image paths
        pred_list = []
        conf_list = []
        label_list = []
        img_path_list = []

    # Check if some of the files in the dataset has already been processed, if so skip them
    df = data_loader.dataset.df.copy()
    existing_files = list(os.listdir(save_dir))
    missing_files = []
    for idx, row in df.iterrows():
        point_cloud_file_name = data_loader.dataset.file_name_from_row(row)
        # replace .laz with .npy
        point_cloud_file_name = point_cloud_file_name.replace(f".{args.pcld_format}", ".npy")
        if point_cloud_file_name not in existing_files:
            missing_files.append(row["id"])
    
    # Filter the dataframe to only include the point clouds which have not been processed yet
    print(f"Found {len(missing_files)} missing files in the dataset.")
    df = df[df["id"].isin(missing_files)]
    data_loader.dataset.df = df
    
    print(f"Number of samples in the {dataset_split} set: ", len(data_loader.dataset))
    model.to(device)
    model.eval()
    torch.backends.cudnn.enabled = True
    with torch.set_grad_enabled(False):
        pbar = tqdm(enumerate(data_loader), total=data_loader.__len__())
        for idx, (fns, data) in pbar:
            
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            target = data['y']
            points = data['x']
            points = points[:, :]
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
            features = model.encoder.forward_cls_feat(data)

            # Save the encoding
            for fn, feature in zip(fns, features):
                point_cloud_file_name = os.path.basename(fn)
                save_path = os.path.join(save_dir, point_cloud_file_name.replace(".laz", ".npy"))
                np.save(save_path, feature.cpu().numpy())
                # print(f"Saved feature encodings to {save_path}")
            

            if args.is_test_performance:
                logits = model(data)
                confusion_matrix.update(logits.argmax(dim=1), target)
                preds = logits.argmax(dim=1)
                pred_list.extend(preds.cpu().numpy())
                
                # Compute confidence scores from the softmax
                conf = torch.nn.functional.softmax(logits, dim=1)
                max_conf = conf.max(dim=1)[0]
                conf_list.extend(max_conf.cpu().numpy())
                
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