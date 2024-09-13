import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, cal_model_parm_nums, load_checkpoint
from openpoints.dataset import build_dataloader_from_cfg


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
    argparse.add_argument('--cfg', type=str, help='config file', default="/workspace/src/cfgs/biovista/pointvector-xl.yaml")
    argparse.add_argument("--source", type=str, help="Path to an image, a directory of images or a csv file with image paths.",
                          default="/workspace/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset_original.csv")
    argparse.add_argument("--model_weights", type=str, help="Path to the model weights file.",
                          default="/workspace/datasets/100_high_and_100_low_HNV-forest-proxy-samples/experiments/BioVista-3D-ALS_pointvector/2024-09-07-15-50_BioVista-3D-ALS_pointvector-s_batch-sz_16_8192_lr_0.001_qb-radius_0.7/checkpoint/2024-09-07-15-50_BioVista-3D-ALS_pointvector-s_batch-sz_16_8192_lr_0.001_qb-radius_0.7_ckpt_best.pth")
    argparse.add_argument("--save_dir", type=str, help="Path to save the encodings.", default=None)
    argparse.add_argument("--shape_size_meters", type=int, help="Shape size in meters.", default=30)
    argparse.add_argument("--batch_size", type=int, help="Batch size for the dataloader.", default=2)
    argparse.add_argument("--num_points", type=int, help="Number of points to sample from the point cloud.", default=8192)
    argparse.add_argument("--num_workers", type=int, help="Number of workers for the dataloader.", default=4)

    args, opts = argparse.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    # Parse the source and validate it
    source = args.source
    if not is_valid_source(source):
        raise ValueError(f"Invalid source {source}.")

    source_is_file = os.path.isfile(source)
    source_is_dir = not source_is_file

    if source_is_dir:
        n_laz_files = len([f for f in os.listdir(source) if f.endswith(".laz")])
        print(f"Found {n_laz_files} .laz files in the directory {source}.")
    if source_is_file and source.endswith(".csv"):
        print(f"Reading image paths from {source}...")
        df = pd.read_csv(source)
        assert "dataset_split" in df.columns, "The csv file must contain a column 'dataset_split'."

        # Get rows with dataset_split == "test"
        df = df[df["dataset_split"] == "test"]
        print("-------------------------------------------------------------------")
        print("WARNING! only using the test samples in the csv file for inference.")
        print("-------------------------------------------------------------------")
        print(f"Found {len(df)} image paths in the csv file.")
    
    cfg.dataset.common.data_root = source

    # Load the model given the config file cfg.yaml
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

    val_loader = build_dataloader_from_cfg(batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split="test",
                                           distributed=False
                                           )
    print("Number of samples in the validation set: ", len(val_loader.dataset))
    model.to(device)
    model.eval()
    torch.backends.cudnn.enabled = True
    with torch.set_grad_enabled(False):
        pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
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
            