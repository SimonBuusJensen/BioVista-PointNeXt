import argparse
import torch
import glob
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

from openpoints.utils import EasyConfig, cal_model_parm_nums, set_random_seed, AverageMeter, ConfusionMatrix, load_checkpoint
from openpoints.optim import build_optimizer_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
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
                        default="/home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/samples.csv")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/samples.csv")
                        # default="/workspace/datasets/samples.csv")
    parser.add_argument('--resnet_weights', type=str, help='ResNet weights file',
                        default="/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth")
    parser.add_argument('--pointvector_weights', type=str, help='PointVector-S weights file',
                        default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
    # parser.add_argument('--mlp_weights', type=str, help='MLP weights file', 
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/MLP-Fusion/Baseline-Frozen/2025-02-20-17-32-55_365_MLP-2D-3D-Fusion_BioVista-MLP-Fusion-Same-Features-v2/mlp_model_81.56_epoch_11.pth")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-02-20-17-32-55_365_MLP-2D-3D-Fusion_BioVista-MLP-Fusion-Same-Features-v2/mlp_model_81.56_epoch_11.pth")
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=5)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2)
    parser.add_argument("--num_workers", type=int, help="The number of threads for the dataloader", default=0)
    parser.add_argument("--fusion_lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--backbone_lr", type=float, help="Learning rate factor for the backbone", default=0)
    
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
    cfg.experiment_dir = os.path.join(os.path.dirname(args.source), "experiments", "2D-3D-Fusion", "MLP-Fusion-Active", cfg.project_name, experiment_name)
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
    logging.info(f'Number of params: {(model_size / 1e6)} M')

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # train_loader.dataset.df = train_loader.dataset.df.sample(100, random_state=cfg.seed)
    
    val_dataset = BioVista2D3D(data_root=args.source, split='val', transform=transform, seed=cfg.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # val_loader.dataset.df = val_loader.dataset.df.sample(100, random_state=cfg.seed)
    
    """
    Training
    """
    # Training arguments
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if cfg.num_workers == 0:
        logging.warning("The number of workers is set to 0, which may slow down the training process.")
    
    cfg.fusion_lr = args.fusion_lr
    assert cfg.fusion_lr is not None, "The fusion learning rate must be provided."
    cfg.backbone_lr = args.backbone_lr
    assert cfg.backbone_lr is not None, "The backbone learning rate must be provided."
    
    # optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    if args.backbone_lr > 0:
        optimizer = torch.optim.AdamW([
            {"params": model.image_backbone.parameters(), "lr": cfg.backbone_lr},
            {"params": model.point_backbone.parameters(), "lr": cfg.backbone_lr},
            {"params": model.fusion_head.parameters(), "lr": cfg.fusion_lr},
        ], weight_decay=1e-2)  # Default weight decay is 1e-2
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.fusion_head.parameters(), "lr": cfg.fusion_lr},
        ], weight_decay=1e-2)
        
        for param in model.image_backbone.parameters():
            param.requires_grad = False
        for param in model.point_backbone.parameters():
            param.requires_grad = False


    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    criterion_args = {'NAME': 'CrossEntropy', 'label_smoothing': 0.2}
    criterion = build_criterion_from_cfg(criterion_args)
    best_val_overall_acc = 0.0
    model.train()  # set model to training mode
    
    for epoch in range(1, cfg.epochs + 1):
        train_pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc=f"Train Epoch [{epoch}/{cfg.epochs}]")
        loss_meter = AverageMeter()
        train_cm = ConfusionMatrix(num_classes=cfg.num_classes)
        for idx, (fn, data) in train_pbar:
            
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
            train_cm.update(logits.argmax(dim=1), target)
            loss_meter.update(loss.item())
        
        # Calculate the accuracy and overall accuracy
        train_loss = loss_meter.avg
        train_macc, train_oacc, accs = train_cm.all_acc()
        fusion_lr = optimizer.param_groups[2]['lr']
        if args.backbone_lr > 0:
            backbone_lr = optimizer.param_groups[0]['lr']
        
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_macc": train_macc,
                "train_oacc": train_oacc,
                "fusion_lr": fusion_lr,
                "backbone_lr": backbone_lr if args.backbone_lr > 0 else 0,
                "epoch": epoch
            })

        # Log the training results
        logging.info(f"Train: Overall acc (%): {train_oacc:.1f}%, Loss: {train_loss:.3f}, fusion_lr: {round(fusion_lr, 7)}, backbone_lr: {round(backbone_lr, 7)}")
        for class_idx in range(train_cm.num_classes):
            class_total_train = train_cm.actual[class_idx].item()
            class_correct_train = train_cm.tp[class_idx].item()
            class_acc_train = (class_correct_train / class_total_train) * 100 if class_total_train > 0 else 0
            logging.info(f"Train: class {train_dataset.classes[class_idx]} (id: {class_idx}) correct: {class_correct_train}/{class_total_train} ({class_acc_train:.1f}%)")
    
    
        """
        VALIDATION
        """
        val_cm = ConfusionMatrix(num_classes=cfg.num_classes)
        is_best = False
        
        with torch.set_grad_enabled(False):
            model.eval()  # set model to eval mode
            val_pred_list = []
            val_conf_list = []
            val_label_list = []
            val_file_path_list = []
            
            val_cm = ConfusionMatrix(num_classes=cfg.num_classes)
            val_pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
            for idx, (fn, data) in val_pbar:
                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                target = data['y']
                points = data['x']
                points = points[:, :cfg.num_points]
                data['pos'] = points[:, :, :3].contiguous()
                data['x'] = points[:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()
                
                # Forward pass
                _2D_features = model.forward_2D_feature_encodings(data['img'])
                _3D_features = model.forward_3D_feature_encodings(data)
                features_2D_3D = torch.cat([_2D_features, _3D_features], dim=1)
                
                logits = model.forward_MLP_predictions(features_2D_3D)
                val_cm.update(logits.argmax(dim=1), target)
                
                # Save the predictions and labels
                val_pred_list.extend(logits.argmax(dim=1).cpu().numpy())

                confidences = torch.nn.functional.softmax(logits, dim=1)
                confidences = torch.max(confidences, 1)[0]

                val_conf_list.extend(confidences.cpu().numpy())
                val_label_list.extend(target.cpu().numpy())
                val_file_path_list.extend(fn)
            
            val_macc, val_overall_acc, accs = val_cm.all_acc()
            
            # Log the validation results
            logging.info(f"Val: Overall acc (%): {val_overall_acc:.1f}%")
            for class_idx in range(val_cm.num_classes):
                class_total_val = val_cm.actual[class_idx].item()
                class_correct_val = val_cm.tp[class_idx].item()
                class_acc_val = (class_correct_val / class_total_val) * 100 if class_total_val > 0 else 0
                logging.info(f"Val: class {val_dataset.classes[class_idx]} (id: {class_idx}) correct: {class_correct_val}/{class_total_val} ({class_acc_val:.1f}%)")
            
            # check if the current model is the best model
            is_best = val_overall_acc > best_val_overall_acc
            if is_best:
                best_val_overall_acc = val_overall_acc
                best_epoch = epoch
                logging.info(f"Best model found at epoch {epoch}, saving model...")
                
                # Delete the previous best model (*.pth file)
                prev_best_model = glob.glob1(cfg.experiment_dir, "multi_modal_fusion_model_*.pth")
                if len(prev_best_model) > 0:
                    logging.info(f"Deleting previous best model: {prev_best_model[0]}")
                    os.remove(os.path.join(cfg.experiment_dir, prev_best_model[0]))
                
                logging.info(f"Saving the best model with overall accuracy: {best_val_overall_acc:.2f}%")
                cur_best_model_fp = os.path.join(cfg.experiment_dir, f"multi_modal_fusion_model_{best_val_overall_acc:.2f}_epoch_{epoch}.pth")
                torch.save(model.state_dict(), cur_best_model_fp)

                # Write the results to a csv file
                pred_label_fp = os.path.join(cfg.experiment_dir, f"val_prediction_labels_epoch_{epoch}_oa_{round(best_val_overall_acc, 1)}.csv")
                with open(pred_label_fp, "w") as f:
                    f.write("image_path,prediction,label,correct,confidence\n")
                    for img_path, pred, label, conf in zip(val_file_path_list, val_pred_list, val_label_list, val_conf_list):
                        f.write(f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
                    # Write overall high, low and total accuracy
                    low_total = val_cm.actual[0].item()
                    low_correct = val_cm.tp[0].item() 
                    low_acc = (low_correct / low_total) * 100 if low_total > 0 else 0
                    f.write(f"Low bio correct,{low_correct},{low_total},{low_acc}\n")
                    high_total = val_cm.actual[1].item()
                    high_correct = val_cm.tp[1].item()
                    high_acc = (high_correct / high_total) * 100 if high_total > 0 else 0
                    f.write(f"High bio correct,{high_correct},{high_total},{high_acc}\n")
                    f.write(f"Overall validation accuracy,{val_cm.tp.sum().item()},{val_cm.actual.sum().item()},{best_val_overall_acc}\n")
                    f.write(f"Mean validation accuracy,,,{val_macc}\n")
                f.close()
                
            if cfg.wandb.use_wandb:
                wandb.save(pred_label_fp)
                wandb.log({
                    "val_acc": val_macc,
                    "val_oacc": val_overall_acc,
                    "best_val_oacc": best_val_overall_acc,
                    "epoch": epoch
                })
        
        scheduler.step(epoch)

    test_dataset = BioVista2D3D(data_root=args.source, split='test', transform=transform, seed=cfg.seed)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # test_loader.dataset.df = test_loader.dataset.df.sample(100, random_state=cfg.seed)
    logging.info("Successfully loaded test dataset. with {} samples".format(len(test_dataset)))

    overall_test_acc = 0.0
    high_correct_test = 0
    low_correct_test = 0
    n_high_bio_samples_test = 0
    n_low_bio_samples_test = 0
    test_pred_list = []
    test_conf_list = []
    test_label_list = []
    test_file_path_list = []

    """
    TESTING
    """
    load_checkpoint(model, pretrained_path=cur_best_model_fp)
    logging.info(f"Loaded the best model from epoch {best_epoch} found during validation.")
    
    model.eval()
    with torch.set_grad_enabled(False):
        for i, (fn, data) in tqdm(enumerate(test_loader), total=test_loader.__len__(), desc=f"Testing:"):

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            labels = data['y'].to(device)
            
            data['pos'] = data['x'][:, :, :3].contiguous()
            data['x'] = data['x'][:, :, :4].transpose(1, 2).contiguous()

            # Forward pass
            _2D_features = model.forward_2D_feature_encodings(data['img'])
            _3D_features = model.forward_3D_feature_encodings(data)
            features_2D_3D = torch.cat([_2D_features, _3D_features], dim=1)
            
            # Save the 2D and 3D encodings
            image_file_name = os.path.basename(fn[0]) + "_30m.png"
            _2D_feature_dir = os.path.join(cfg.experiment_dir, "resnet_encodings")
            if not os.path.exists(_2D_feature_dir):
                os.makedirs(_2D_feature_dir, exist_ok=True)
            _2D_feature_fp = os.path.join(_2D_feature_dir, image_file_name.replace(".png", ".npy"))
            
            if not os.path.exists(_2D_feature_fp):
                np.save(_2D_feature_fp, _2D_features.cpu().numpy())
            
            point_cloud_file_name = os.path.basename(fn[0]) + "_30m.npz"
            _3D_feature_dir = os.path.join(cfg.experiment_dir, "pointvector_encodings")
            if not os.path.exists(_3D_feature_dir):
                os.makedirs(_3D_feature_dir, exist_ok=True)
            _3D_feature_fp = os.path.join(_3D_feature_dir, point_cloud_file_name.replace(".npz", ".npy"))
            
            if not os.path.exists(_3D_feature_fp):
                np.save(_3D_feature_fp, _3D_features.cpu().numpy())
            
            outputs = model.forward_MLP_predictions(features_2D_3D)
            _, preds = torch.max(outputs, 1)
            # Calculate the confidence scores between 0-100% for the predictions
            confidences = torch.nn.functional.softmax(outputs, dim=1)
            confidences = torch.max(confidences, 1)[0]

            overall_test_acc += torch.sum(preds == labels.data)
            high_correct_test += torch.sum((preds == labels.data) & (labels == 1))
            low_correct_test += torch.sum((preds == labels.data) & (labels == 0))

            n_high_bio_samples_test += torch.sum(labels == 1)
            n_low_bio_samples_test += torch.sum(labels == 0)

            # Append the predictions and labels to the lists
            test_pred_list.extend(preds.cpu().numpy())
            test_label_list.extend(labels.cpu().numpy())
            test_file_path_list.extend(fn)
            # Append the confidence scores as float with 2 decimals
            test_conf_list.extend(confidences.cpu().detach().numpy())

    # Calculate the overall test accuracy
    overall_test_acc = round(overall_test_acc.item() / len(test_dataset) * 100, 2)
    if n_high_bio_samples_test.item() == 0:
        overall_val_acc_high = 0.0
    else:
        overall_val_acc_high = round(high_correct_test.item() / n_high_bio_samples_test.item() * 100, 2)

    if n_low_bio_samples_test.item() == 0:
        overall_val_acc_low = 0.0
    else:
        overall_val_acc_low = round(low_correct_test.item() / n_low_bio_samples_test.item() * 100, 2)

    # Write the image_paths, predictions and labels to a csv file
    pred_label_fp = os.path.join(cfg.experiment_dir, f"test_prediction_labels.csv")
    with open(pred_label_fp, "w") as f:
        f.write("image_path,prediction,label,correct,confidence\n")
        for img_path, pred, label, conf in zip(test_file_path_list, test_pred_list, test_label_list, test_conf_list):
            f.write(f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
        # Write overall high, low and total accuracy
        f.write(f"Low bio correct,{low_correct_test.item()},{n_low_bio_samples_test.item()},{overall_val_acc_low}\n")
        f.write(f"High bio correct,{high_correct_test.item()},{n_high_bio_samples_test.item()},{overall_val_acc_high}\n")
        f.write(f"Overall test accuracy,{low_correct_test.item() + high_correct_test.item()},{len(test_dataset)},{round(overall_test_acc, 2)}\n")
        f.write(f"Mean test accuracy,,,{round((overall_val_acc_low + overall_val_acc_high) / 2, 2)}\n")
    f.close()
    
    if args.use_wandb:
        wandb.log({
            "test_macc": round((overall_val_acc_low + overall_val_acc_high) / 2, 2),
            "test_oacc": overall_test_acc,
            "test_low_bio_acc": overall_val_acc_low,
            "test_high_bio_acc": overall_val_acc_high
        })
