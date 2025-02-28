import argparse
import os
import sys
import wandb
import logging 
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from openpoints.utils import EasyConfig, set_random_seed, cal_model_parm_nums
from openpoints.dataset import build_dataloader_from_cfg
from train_classifier import str2bool
from fusion_classifier.MLPModel import MLPModel
from fusion_classifier.FeatureDataset import FeatureDataset


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training a BioVista dataset classifier consisting of a PointVector and a ResNet Model which are then fused together with a MLP for fusion')
    # Dataset settings
    parser.add_argument("--csv", type=str, help="Path to a csv file containing the name of the training and validation data files.",
                        default="/workspace/datasets/samples.csv")
    # parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    # parser.add_argument("--num_workers", type=int, help="Number of workers for the dataloader", default=4)
    # parser.add_argument("--num_points", type=int, help="Number of points to sample from the point cloud", default=16384)

    # Training settings
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)


    # PointVector Settings
    parser.add_argument('--pointvector_cfg', type=str, help='config file', 
                        # default="cfgs/biovista_2D_3D/pointvector-s.yaml")
                        default="/workspace/src/cfgs/biovista_2D_3D/pointvector-s.yaml")
    parser.add_argument("--features_dir_2d", type=str, help="Path to a directory containing the 2D features of the images.",
                        default="/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/resnet_encodings/")
                        # default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/resnet_encodings/")
    parser.add_argument("--pointvector_model_weights", type=str, help="Path to the model weights file.",
                        default="")

    # ResNet Settings:
    parser.add_argument("--features_dir_3d", type=str, help="Path to a directory containing the 3D features of the point clouds.",
                        default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/pointvector_encodings/")
                        # default="/home/simon/data/BioVista/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/pointvector_encodings/")
    parser.add_argument("--resnet_model_weights", type=str, help="Path to the model weights file.",
                        default="")

    # MLP Settings
    parser.add_argument("--mlp_model_weights", type=str, help="Path to the model weights file.",
                        default="")
    

    # General Project Settings:
    parser.add_argument("--project_name", type=str, help="Weights and biases project name", default="BioVista-MLP-Fusion-2D-3D-Active-Weights-Test-Version")
    parser.add_argument("--mode", type=str, help="Mode (train or test)", default="test")
    parser.add_argument("--is_active_weights", type=str2bool, help="Whether to freeze the weights of the PointVector and ResNet models", default=False)
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument("--use_wandb", type=str2bool, help="Whether to log to weights and biases", default=True)

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    assert os.path.exists(args.pointvector_cfg), "The PointVector config file does not exist."
    cfg.load(args.pointvector_cfg, recursive=True)
    cfg.update(opts)

    # Setup the csv file
    csv_file = args.csv
    assert os.path.exists(csv_file), "The csv file does not exist."
    assert csv_file.endswith(".csv"), "The csv file must have a .csv extension."
    cfg.dataset.common.data_root = csv_file

    # Setup project name and experiment name
    assert args.project_name is not None
    assert isinstance(args.project_name, str), "The project_name must be a string."
    cfg.project_name = args.project_name
    date_now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{date_now_str}-{cfg.project_name}"

    # Setup output dir for the experiment to save log, models, test results, etc.
    cfg.experiment_dir = os.path.join(os.path.dirname(csv_file), "experiments", "2D-3D-Fusion", "MLP-Fusion", cfg.project_name, experiment_name)
    print(f"Output directory: {cfg.experiment_dir}")
    os.makedirs(cfg.experiment_dir, exist_ok=True)

    # Init logger
    log_file = os.path.join(cfg.experiment_dir, f"{experiment_name}.log")
    setup_logger(log_file)    

    # Set the seed
    if args.seed is not None:
        assert 1 <= args.seed <= 10000, "The seed must be between 1 and 10000."
        assert isinstance(args.seed, int), "The seed must be an integer."
        cfg.seed = args.seed
    else:
        cfg.seed = np.random.randint(1, 10000)
    logging.info(f"Seed: {cfg.seed}")

    set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    # Set the mode
    cfg.mode = args.mode
    assert cfg.mode in ["train", "test"]
    logging.info(f"Mode: {cfg.mode}")

    # Set with or without active learning (is_active_weights)
    cfg.is_active_weights = args.is_active_weights
    assert isinstance(cfg.is_active_weights, bool), "The is_active_weights must be a boolean."
    logging.info(f"Active weights: {cfg.is_active_weights}")
    
    # Setup wandb
    assert isinstance(args.use_wandb, bool), "The use_wandb must be a boolean."
    if args.use_wandb and cfg.mode == "train":
        cfg.wandb.use_wandb = True
        cfg.wandb.project = cfg.project_name
        wandb.init(project=cfg.wandb.project, name=experiment_name)
        wandb.config.update(args)
        wandb.save(log_file)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Setup dataloaders
    if cfg.mode == "test":
        test_dataset = FeatureDataset(csv_file, 
                                      feature_dir_2d=args.features_dir_2d,
                                      feature_dir_3d=args.features_dir_3d,
                                      data_split="test")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        logging.info(f"Test dataset: {len(test_dataset)} samples.")
    else: # mode: train
        cfg.dataset.train.num_points = cfg.num_points
        train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='train'
                                                )
        logging.info(f"length of training dataset: {len(train_loader.dataset)}")
        cfg.dataset.val.num_points = cfg.num_points
        val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='val'
                                                )
        logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
        cfg.dataset.test.num_points = cfg.num_points
        cfg.dataset.test.seed = cfg.seed
        test_loader = build_dataloader_from_cfg(1,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='test'
                                                )


    # Assert the features_dir_2d and features_dir_3d exists and are none empty directories in is_active_weights is False
    if not cfg.is_active_weights:
        assert args.features_dir_2d is not None and args.features_dir_3d is not None
        assert os.path.exists(args.features_dir_2d), "The 2D features directory does not exist."
        assert os.path.exists(args.features_dir_3d), "The 3D features directory does not exist."
        assert len(os.listdir(args.features_dir_2d)) == 44378, "The 2D features does not contain 44378 files."
        assert len(os.listdir(args.features_dir_3d)) == 44378, "The 3D features does not contain 44378 files."
        logging.info(f"2D features directory: {args.features_dir_2d}")
        logging.info(f"3D features directory: {args.features_dir_3d}")

        # Init model
        model = MLPModel()
        num_params = cal_model_parm_nums(model)
        logging.info(f"Model: {model}: {num_params} parameters.")


        """
        TESTING MLP MODEL
        """

        if cfg.mode == "test":
            assert args.mlp_model_weights is not None
            assert os.path.exists(args.mlp_model_weights), "The MLP model weights file does not exist."
            logging.info(f"Loading MLP model weights from: {args.mlp_model_weights}")
            model.load_state_dict(torch.load(args.mlp_model_weights))
            model.eval()
            model.to(device)

            overall_accuracy_test = 0
            high_correct = 0
            low_correct = 0
            n_high_bio_samples = 0
            n_low_bio_samples = 0

            pred_list = []
            label_list = []
            file_name_list = []
            conf_list = []

            with torch.no_grad():
                for fn, X_test_batch, y_test_batch in tqdm(test_loader, desc="Evaluating on test set"):

                    # Move tensors to the same device
                    X_test_batch = X_test_batch.to(device)
                    y_test_batch = y_test_batch.to(device)

                    outputs = model(X_test_batch)
                    confidences = torch.nn.functional.softmax(outputs, dim=1)
                    confidences = torch.max(confidences, 1)[0]

                    _, preds = torch.max(outputs, 1)
                    labels = torch.max(y_test_batch, 1)[1]
                    overall_accuracy_test += torch.sum(preds == labels).item()

                    high_correct += torch.sum((preds == labels) & (labels == 1))
                    low_correct += torch.sum((preds == labels) & (labels == 0))

                    n_high_bio_samples += torch.sum(labels == 1)
                    n_low_bio_samples += torch.sum(labels == 0)

                    # Append results
                    pred_list.extend(preds.cpu().numpy())
                    label_list.extend(labels.cpu().numpy())
                    file_name_list.extend(fn)
                    conf_list.extend(confidences.cpu().detach().numpy())

        overall_accuracy_test = overall_accuracy_test / len(test_dataset) * 100

        if n_high_bio_samples.item() == 0:
            overall_val_acc_high = 0.0
        else:
            overall_val_acc_high = round(
                high_correct.item() / n_high_bio_samples.item() * 100, 2)

        if n_low_bio_samples.item() == 0:
            overall_val_acc_low = 0.0
        else:
            overall_val_acc_low = round(
                low_correct.item() / n_low_bio_samples.item() * 100, 2)

        results_file = os.path.join(cfg.experiment_dir, "test_results.csv")
        with open(results_file, "w") as f:
            f.write("file_name,prediction,label,correct,confidence\n")
            for img_path, pred, label, conf in zip(file_name_list, pred_list, label_list, conf_list):
                f.write(f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf * 100, 0)}\n")

            # Write overall high, low and total accuracy
            f.write(f"Low bio correct,{low_correct.item()},{n_low_bio_samples.item()},{overall_val_acc_low}\n")
            f.write(f"High bio correct,{high_correct.item()},{n_high_bio_samples.item()},{overall_val_acc_high}\n")
            f.write(f"Overall validation accuracy,{low_correct.item()+high_correct.item()},{len(test_dataset)},{overall_accuracy_test}\n")
            mean_val_acc = (overall_val_acc_low + overall_val_acc_high) / 2
            f.write(f"Mean validation accuracy,,,{mean_val_acc}")
            print(f"Evaluation results saved to {results_file}")
            f.close()

            if args.wandb:
                wandb.log({
                "overall_accuracy_test": overall_accuracy_test,
                "mean_accuracy_test": mean_val_acc,
                "high_bio_acc_test": overall_val_acc_high,
                "low_bio_acc_test": overall_val_acc_low
            })


