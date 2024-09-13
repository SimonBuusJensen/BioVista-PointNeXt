import os
import argparse
import pandas as pd
from openpoints.models import build_model_from_cfg
from openpoints.utils import cal_model_parm_nums
from openpoints.utils import EasyConfig


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
                          default="/media/simon/Elements/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset_original.csv")
    argparse.add_argument("--model_weights", type=str, help="Path to the model weights file.",
                          default="/media/simon/Elements/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/experiments/BioVista-2D-Orthophotos/BioVista-Orthophotos-Shape-Sizes/2024-08-27-13-30_BioVista-Orthophotos-Shape-Size_resnet50_640px_30m_circles/2024-08-27-13-30_resnet50_epoch_60_acc_78.46.pth")
    argparse.add_argument("--save_dir", type=str, help="Path to save the encodings.", default=None)
    argparse.add_argument("--shape_size_meters", type=int, help="Shape size in meters.", default=30)
    argparse.add_argument("--batch_size", type=int, help="Batch size for the dataloader.", default=4)
    argparse.add_argument("--num_workers", type=int, help="Number of workers for the dataloader.", default=0)

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

    
    model = build_model_from_cfg(cfg.model)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))