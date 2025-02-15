import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, cal_model_parm_nums, load_checkpoint, ConfusionMatrix, set_random_seed
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


if __name__ == "__main__":

    # Parse arguments
    argparse = argparse.ArgumentParser(description="Generate feature encodings for 3D Point clouds using a trained Point Vector Model")
    argparse.add_argument('--cfg', type=str, help='config file', default="/workspace/src/cfgs/biovista/pointvector-s.yaml")
    argparse.add_argument("--source", type=str, help="Path to an image, a directory of images or a csv file with image paths.",
                        #   default="/workspace/datasets/high_and_low_HNV-forest-proxy-test-dataset/high_and_low_HNV-forest-proxy-polygon-test-dataset_30_m_circles_dataset_wihtout_empty_clouds.csv")
                          default="/workspace/datasets/samples.csv")
    argparse.add_argument("--model_weights", type=str, help="Path to the model weights file.",
                          default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-pointc-cloud-PointVector/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
    argparse.add_argument("--batch_size", type=int, help="Batch size for the dataloader.", default=1)
    argparse.add_argument("--qb_radius", type=float, help="Query ball radius", default=0.65)
    argparse.add_argument("--qb_radius_scaling", type=float, help="Radius scaling factor", default=1.5)
    argparse.add_argument("--num_points", type=int, help="Number of points to sample from the point cloud.", default=16384)
    argparse.add_argument("--num_workers", type=int, help="Number of workers for the dataloader.", default=4)
    argparse.add_argument("--pcld_format", type=str, help="File format of the dataset.", default="npz")
    argparse.add_argument("--channels", type=str, help="Channels to use, x, y, z, h (height) and/or i (intensity)", default="xyzh")
    argparse.add_argument("--with_normalize_gravity_dim", type=str2bool, help="Whether to normalize the gravity dimension", default=False)
    argparse.add_argument("--seed", type=int, help="Random seed", default=9447)

    args, opts = argparse.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    # Set the seed
    set_random_seed(args.seed, deterministic=cfg.deterministic)

    # Parse the source and validate it
    source = args.source 
    
    assert source.endswith(".csv"), f"Source {source} must be a .csv file."

    print(f"Reading image paths from {source}...")
    df = pd.read_csv(source)
    dataset_split = "test"
    assert "dataset_split" in df.columns, "The csv file must contain a column 'dataset_split'."

    # Get rows with dataset_split == "test"
    df = df[df["dataset_split"] == dataset_split]
    print("-------------------------------------------------------------------")
    print("WARNING! only using the test samples in the csv file for inference.")
    print("-------------------------------------------------------------------")
    print(f"Found {len(df)} image paths in the csv file.")
    
     # Set the number of points in the point cloud and the channels
    cfg.dataset.common.data_root = source
    cfg.dataset.common.num_points = args.num_points
    cfg.dataset.common.channels = args.channels
    assert args.channels in ["xyz", "xyzi", "xyzh", "xyzhi", "xyzih"], "Channels must be one of xyz, xyzi, xyzh, xyzhi, xyzih"

    # Set the Query ball parameters
    cfg.model.encoder_args.radius = args.qb_radius
    cfg.model.encoder_args.radius_scaling = args.qb_radius_scaling
    cfg.model.encoder_args.in_channels = len(args.channels)

    if args.with_normalize_gravity_dim:
        cfg.datatransforms.kwargs.normalize_gravity_dim = True
    else:
        cfg.datatransforms.kwargs.normalize_gravity_dim = False

    # Load the model given the config file cfg.yaml
    model = build_model_from_cfg(cfg.model)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # Load the model weights
    model_weights = args.model_weights
    assert os.path.exists(model_weights), f"Model weights file {model_weights} does not exist."
    test_dir = os.path.join(os.path.dirname(os.path.dirname(model_weights)))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # test mode
    print("Loading model weights from: " + model_weights + "...")
    epoch, best_val = load_checkpoint(model, pretrained_path=model_weights)
    # Set model to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.model.in_channels = cfg.model.encoder_args.in_channels
    print(f"Loaded model weights from epoch {epoch} with best validation accuracy {best_val}")

    # Setup the dataset and data loader
    batch_size = args.batch_size
    assert batch_size > 0, "Batch size must be greater than 0."
    num_workers = args.num_workers
    cfg.dataloader.num_workers = num_workers
    if num_workers == 0:
        print("Warning: num_workers is set to 0. This might slow down the training process.")

    # Assert pcld_format is either npz or laz
    assert args.pcld_format in ["npz", "laz"], f"Point cloud format {args.pcld_format} is not supported."
    cfg.dataset.common.format = args.pcld_format

    test_loader = build_dataloader_from_cfg(batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split=dataset_split
                                           )
    
    print(f"Number of samples in the {dataset_split} set: ", len(test_loader.dataset))
    model.to(device)
    model.eval()
    torch.backends.cudnn.enabled = True
    with torch.set_grad_enabled(False):

        pred_list = []
        conf_list = []
        label_list = []
        img_path_list = []
        test_cm = ConfusionMatrix(num_classes=cfg.num_classes)

        pbar = tqdm(enumerate(test_loader), total=test_loader.__len__())
        for idx, (fns, data) in pbar:
            
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            target = data['y']
            points = data['x']

            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
            logits = model(data)
            test_cm.update(logits.argmax(dim=1), target)

            # Save the predictions and labels
            pred_list.extend(logits.argmax(dim=1).cpu().numpy())

            confidences = torch.nn.functional.softmax(logits, dim=1)
            confidences = torch.max(confidences, 1)[0]

            conf_list.extend(confidences.cpu().numpy())
            label_list.extend(target.cpu().numpy())
            img_path_list.extend(fns)

        tp, count = test_cm.tp, test_cm.count
        test_macc, test_oa, _ = test_cm.cal_acc(tp, count)
        pred_label_fp = os.path.join(test_dir, f"test_prediction_labels.csv")
        with open(pred_label_fp, "w") as f:
            f.write("image_path,prediction,label,correct,confidence\n")
            for img_path, pred, label, conf in zip(img_path_list, pred_list, label_list, conf_list):
                f.write(
                    f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
            # Write overall high, low and total accuracy
            low_total = test_cm.actual[0].item()
            low_correct = test_cm.tp[0].item() 
            low_acc = (low_correct / low_total) * 100 if low_total > 0 else 0
            f.write(f"Low bio correct,{low_correct},{low_total},{low_acc}\n")
            high_total = test_cm.actual[1].item()
            high_correct = test_cm.tp[1].item()
            high_acc = (high_correct / high_total) * 100 if high_total > 0 else 0
            f.write(f"High bio correct,{high_correct},{high_total},{high_acc}\n")
            f.write(f"Overall test accuracy,{test_cm.tp.sum().item()},{test_cm.actual.sum().item()},{test_oa}\n")
            f.write(f"Mean test accuracy,{test_macc}\n")
        f.close()