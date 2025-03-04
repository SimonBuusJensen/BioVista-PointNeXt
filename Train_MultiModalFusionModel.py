import argparse
import torch
import os
import logging
import sys
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from datetime import datetime
from tqdm import tqdm

from openpoints.utils import EasyConfig, cal_model_parm_nums, set_random_seed, AverageMeter, ConfusionMatrix
from openpoints.optim import build_optimizer_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.dataset import BioVista2D3D
from Test_MultiModalFusionModel import MultiModalFusionModel
from train_classifier import str2bool

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
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, help='config file',
                        # default="/workspace/src/cfgs/biovista_2D_3D/pointvector-s.yaml")
                        default="cfgs/biovista/pointvector-s.yaml")
    parser.add_argument("--source", type=str, help="Path to an image, a directory of images or a csv file with image paths.",
                        default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/samples.csv")
                        # default="/workspace/datasets/samples.csv")
    parser.add_argument('--resnet_weights', type=str, help='ResNet weights file',
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth")
                        default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth")
    parser.add_argument('--pointvector_weights', type=str, help='PointVector-S weights file',
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
    # parser.add_argument('--mlp_weights', type=str, help='MLP weights file', 
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/MLP-Fusion/Baseline-Frozen/2025-02-20-17-32-55_365_MLP-2D-3D-Fusion_BioVista-MLP-Fusion-Same-Features-v2/mlp_model_81.56_epoch_11.pth")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-02-20-17-32-55_365_MLP-2D-3D-Fusion_BioVista-MLP-Fusion-Same-Features-v2/mlp_model_81.56_epoch_11.pth")
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=10)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2)
    parser.add_argument("--num_workers", type=int, help="The number of threads for the dataloader", default=0)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    
    # General arguments
    parser.add_argument("--use_wandb", type=str2bool, help="Whether to log to weights and biases", default=True)
    parser.add_argument("--project_name", type=str, help="Weights and biases project name", default="BioVista-Multimodal-Fusion-Active-Weights-Test")
    
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    
    # Set the seed
    if args.seed is not None:
        cfg.seed = args.seed
    else:
        cfg.seed = np.random.randint(1, 10000)
        
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True    
    
    # Setup project name and experiment name
    assert args.project_name is not None
    assert isinstance(args.project_name, str), "The project_name must be a string."
    cfg.project_name = args.project_name
    date_now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{date_now_str}-{cfg.project_name}"
    
    # Setup output dir for the experiment to save log, models, test results, etc.
    cfg.experiment_dir = os.path.join(os.path.dirname(args.source), "experiments", "2D-3D-Fusion", "MLP-Fusion", cfg.project_name, experiment_name)
    print(f"Output directory: {cfg.experiment_dir}")
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    
    # Init logger
    log_file = os.path.join(cfg.experiment_dir, f"{experiment_name}.log")
    setup_logger(log_file) 

    
    # Setup wandb
    assert isinstance(args.use_wandb, bool), "The use_wandb must be a boolean."
    if args.use_wandb and cfg.mode == "train":
        cfg.wandb.use_wandb = True
        cfg.wandb.project = cfg.project_name
        wandb.init(project=cfg.wandb.project, name=experiment_name)
        wandb.config.update(args)
        wandb.save(log_file)
        
    # Model arguments
    cfg.model.encoder_args.in_channels = 4  # xyzh
    cfg.model.encoder_args.radius = 0.65
    cfg.model.encoder_args.radius_scaling = 1.5

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = build_model_from_cfg(cfg.model).to(device)
    model = MultiModalFusionModel()
    model_size = cal_model_parm_nums(model)
    # print(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # Check model weights
    resnet_model_weights = args.resnet_weights
    pointvector_weights = args.pointvector_weights
    
    assert resnet_model_weights is not None, "ResNet model weights must be provided."
    assert os.path.exists(resnet_model_weights), "ResNet model weights not found."
    assert pointvector_weights is not None, "PointVector-S model weights must be provided."
    assert os.path.exists(pointvector_weights), "PointVector-S model weights not found."
    
    # Test if we can load ResNet model weights
    # resnet_model_weights = "/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/2025-01-21-15-02-20_resnet18_epoch_9_acc_79.25.pth"
    model.load_weights(resnet_weights_path=resnet_model_weights, pointvector_weights_path=pointvector_weights, mlp_weights_path=None, map_location=device)
    model.to(device)

    from torchvision.transforms import Compose
    from openpoints.transforms import PointsToTensor, PointCloudXYZAlign
    transform = Compose([PointsToTensor(), PointCloudXYZAlign(normalize_gravity_dim=False)])
    train_dataset = BioVista2D3D(data_root=args.source, split='train', transform=transform, seed=cfg.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_loader.dataset.df = train_loader.dataset.df.sample(100, random_state=cfg.seed)
    
    val_dataset = BioVista2D3D(data_root=args.source, split='val', transform=transform, seed=cfg.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    """
    Training
    """
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    criterion_args = {'NAME': 'CrossEntropy', 'label_smoothing': 0.2}
    criterion = build_criterion_from_cfg(criterion_args)
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)

    model.train()  # set model to training mode
    
    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc=f"Train Epoch [{epoch}/{cfg.epochs}]")
        for idx, (fn, data) in pbar:
            
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
        
            points = data['x']
            target = data['y']
        
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()
            
            # Forward pass
            _2D_features = model.forward_2D_feature_encodings(data['img'])
            _3D_features = model.forward_3D_feature_encodings(data)
            features_2D_3D = torch.cat([_2D_features, _3D_features], dim=1)
            
            logits = model.forward_MLP_predictions(features_2D_3D)
            loss = criterion(logits, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)    
            optimizer.step()
            model.zero_grad()
        
            # update confusion matrix
            cm.update(logits.argmax(dim=1), target)
            loss_meter.update(loss.item())
        
        # Calculate the accuracy and overall accuracy
        train_loss = loss_meter.avg
        train_macc, train_oa, accs = cm.all_acc()
        lr = optimizer.param_groups[0]['lr']
        
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_macc,
                "train_oa": train_oa,
                "lr": lr,
                "epoch": epoch
            })

        

    # test_dataset = BioVista2D3D(data_root=args.source, split='test', transform=transform, seed=cfg.seed)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # test_loader.dataset.df = test_loader.dataset.df.sample(100, random_state=cfg.seed)
    # print("Successfully loaded test dataset. with {} samples".format(len(test_dataset)))

    # test_acc = 0.0
    # high_correct = 0
    # low_correct = 0
    # n_high_bio_samples = 0
    # n_low_bio_samples = 0
    # pred_list = []
    # conf_list = []
    # label_list = []
    # file_path_list = []

    # """
    # TESTING
    # """
    # model.eval()
    # with torch.set_grad_enabled(False):
    #     for i, (fn, data) in tqdm(enumerate(test_loader), total=test_loader.__len__()):

    #         for key in data.keys():
    #             data[key] = data[key].cuda(non_blocking=True)

    #         labels = data['y'].to(device)
            
    #         data['pos'] = data['x'][:, :, :3].contiguous()
    #         data['x'] = data['x'][:, :, :4].transpose(1, 2).contiguous()

    #         # Forward pass
    #         _2D_features = model.forward_2D_feature_encodings(data['img'])
    #         _3D_features = model.forward_3D_feature_encodings(data)
    #         features_2D_3D = torch.cat([_2D_features, _3D_features], dim=1)
            
    #         # Save the 2D and 3D encodings
    #         image_file_name = os.path.basename(fn[0]) + "_30m.png"
    #         _2D_feature_dir = os.path.join(os.path.dirname(resnet_model_weights), "resnet_encodings")
    #         if not os.path.exists(_2D_feature_dir):
    #             os.makedirs(_2D_feature_dir, exist_ok=True)
    #         _2D_feature_fp = os.path.join(_2D_feature_dir, image_file_name.replace(".png", ".npy"))
            
    #         if not os.path.exists(_2D_feature_fp):
    #             np.save(_2D_feature_fp, _2D_features.cpu().numpy())
            
    #         point_cloud_file_name = os.path.basename(fn[0]) + "_30m.npz"
    #         _3D_feature_dir = os.path.join(os.path.dirname(os.path.dirname(pointvector_weights)), "pointvector_encodings")
    #         if not os.path.exists(_3D_feature_dir):
    #             os.makedirs(_3D_feature_dir, exist_ok=True)
    #         _3D_feature_fp = os.path.join(_3D_feature_dir, point_cloud_file_name.replace(".npz", ".npy"))
            
    #         if not os.path.exists(_3D_feature_fp):
    #             np.save(_3D_feature_fp, _3D_features.cpu().numpy())
            
    #         outputs = model.forward_MLP_predictions(features_2D_3D)
    #         _, preds = torch.max(outputs, 1)
    #         # Calculate the confidence scores between 0-100% for the predictions
    #         confidences = torch.nn.functional.softmax(outputs, dim=1)
    #         confidences = torch.max(confidences, 1)[0]

    #         test_acc += torch.sum(preds == labels.data)
    #         high_correct += torch.sum((preds == labels.data) & (labels == 1))
    #         low_correct += torch.sum((preds == labels.data) & (labels == 0))

    #         n_high_bio_samples += torch.sum(labels == 1)
    #         n_low_bio_samples += torch.sum(labels == 0)

    #         # Append the predictions and labels to the lists
    #         pred_list.extend(preds.cpu().numpy())
    #         label_list.extend(labels.cpu().numpy())
    #         file_path_list.extend(fn)
    #         # Append the confidence scores as float with 2 decimals
    #         conf_list.extend(confidences.cpu().detach().numpy())

    # # Calculate the overall validation accuracy
    # overall_val_acc = round(test_acc.item() / len(test_dataset) * 100, 2)
    # if n_high_bio_samples.item() == 0:
    #     overall_val_acc_high = 0.0
    # else:
    #     overall_val_acc_high = round(
    #         high_correct.item() / n_high_bio_samples.item() * 100, 2)

    # if n_low_bio_samples.item() == 0:
    #     overall_val_acc_low = 0.0
    # else:
    #     overall_val_acc_low = round(
    #         low_correct.item() / n_low_bio_samples.item() * 100, 2)

    # # Write the image_paths, predictions and labels to a csv file
    # pred_label_fp = os.path.join(
    #     output_dir, f"prediction_labels_from_the_mulitmodal_fusion_model.csv")
    # with open(pred_label_fp, "w") as f:
    #     f.write("image_path,prediction,label,correct,confidence\n")
    #     for img_path, pred, label, conf in zip(file_path_list, pred_list, label_list, conf_list):
    #         f.write(
    #             f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
    #     # Write overall high, low and total accuracy
    #     f.write(
    #         f"Low bio correct,{low_correct.item()},{n_low_bio_samples.item()},{overall_val_acc_low}\n")
    #     f.write(
    #         f"High bio correct,{high_correct.item()},{n_high_bio_samples.item()},{overall_val_acc_high}\n")
    #     f.write(
    #         f"Overall test accuracy,{test_acc},{len(test_dataset)},{overall_val_acc}\n")
    #     f.write(
    #         f"Mean test accuracy,,,{(overall_val_acc_low + overall_val_acc_high) / 2}\n")
    # f.close()
