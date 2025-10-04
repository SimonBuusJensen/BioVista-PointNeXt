import argparse
import glob
import logging
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from Test_MultiModalFusionModel import MultiModalFusionModel
from openpoints.dataset import BioVista2D3D
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.transforms import PointsToTensor, PointCloudXYZAlign
from openpoints.utils import EasyConfig, cal_model_parm_nums, set_random_seed, AverageMeter, ConfusionMatrix, \
    load_checkpoint
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


def calculate_class_weights(labels):
    # Count the occurrence of each class
    class_counts = Counter(labels)
    total_count = sum(class_counts.values())

    # Inverse of frequency
    class_weights = {cls: total_count /
                          count for cls, count in class_counts.items()}

    # Convert to a tensor
    class_weights_tensor = torch.tensor(
        [class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float)
    return class_weights_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, help='config file', default="cfgs/biovista/pointvector-s.yaml")
    parser.add_argument("--source", type=str,
                        help="Path to csv file with image paths.")
    parser.add_argument('--resnet_weights', type=str, help='ResNet weights file', default=None)
    parser.add_argument('--pointvector_weights', type=str, help='PointVector-S weights file', default=None)
    parser.add_argument('--orthophoto_channels', type=str, help='RGB, NGB, RGBN', default="NRG")
    parser.add_argument('--seed', type=int, help='Random seed', default=None)

    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=5)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2)
    parser.add_argument("--grad_accum_steps", type=int, help="Gradient accumulation steps (>=1)", default=1)
    parser.add_argument("--in_memory", help="Cache whole dataset", type=str2bool, default=False)
    parser.add_argument("--num_workers", type=int, help="The number of threads for the dataloader", default=2)
    parser.add_argument("--fusion_lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--backbone_lr", type=float, help="Learning rate factor for the backbone", default=0.000001)
    parser.add_argument("--with_shortcut_fusion", type=str2bool, help="Whether to use shortcut fusion", default=False)
    parser.add_argument("--with_class_weights", type=str2bool, help="Whether to use class weighted loss", default=True)

    # General arguments
    parser.add_argument("--use_wandb", type=str2bool, help="Whether to log to weights and biases", default=True)
    parser.add_argument("--project_name", type=str, help="Weights and biases project name",
                        default="BioVista-Multimodal-Fusion-Active-Weights-Test")

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.batch_size = args.batch_size  # override

    experiment_id = np.random.randint(1000, 9999)

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
    experiment_name = f"{date_now_str}-{experiment_id}-{cfg.project_name}"

    # Setup output dir for the experiment to save log, models, test results, etc.
    cfg.experiment_dir = os.path.join(os.path.dirname(args.source), "experiments", "2D-3D-Fusion",
                                      "MLP-Fusion-Active-and-Frozen-Backbones", cfg.project_name, experiment_name)
    os.makedirs(cfg.experiment_dir, exist_ok=True)

    # Init logger
    log_file = os.path.join(cfg.experiment_dir, f"{experiment_name}.log")
    setup_logger(log_file)

    transform = Compose([PointsToTensor(), PointCloudXYZAlign(normalize_gravity_dim=False)])
    train_dataset = BioVista2D3D(
        data_root=args.source, split='train', transform=transform, orthophoto_channels=args.orthophoto_channels,
        in_memory=args.in_memory
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)
    # train_loader.dataset.df = train_loader.dataset.df.sample(200, random_state=cfg.seed)

    val_dataset = BioVista2D3D(
        data_root=args.source, split='val', transform=transform, orthophoto_channels=args.orthophoto_channels,
        in_memory=args.in_memory
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    # val_loader.dataset.df = val_loader.dataset.df.sample(200, random_state=cfg.seed)
    cfg.num_classes = train_dataset.num_classes

    cfg.fusion_lr = args.fusion_lr
    assert cfg.fusion_lr is not None, "The fusion learning rate must be provided."
    cfg.backbone_lr = args.backbone_lr
    assert cfg.backbone_lr is not None, "The backbone learning rate must be provided."

    # Model arguments
    with_shortcut_fusion = args.with_shortcut_fusion
    assert isinstance(with_shortcut_fusion, bool), "The with_shortcut_fusion must be a boolean."
    pts_channel = 4
    img_channel = len(args.orthophoto_channels)
    cfg.model.encoder_args.in_channels = pts_channel  # xyzh
    cfg.model.encoder_args.radius = 0.65
    cfg.model.encoder_args.radius_scaling = 1.5

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = build_model_from_cfg(cfg.model).to(device)
    model = MultiModalFusionModel(
        num_classes=train_dataset.num_classes, img_channel=img_channel,
        pts_channel=pts_channel, with_shortcut_fusion=with_shortcut_fusion,
        freeze_backbone=cfg.backbone_lr == 0.0
    )
    model_size = cal_model_parm_nums(model)
    logging.info(f'Number of params: {(model_size / 1e6)} M')

    # Check model weights
    resnet_model_weights = args.resnet_weights
    pointvector_weights = args.pointvector_weights

    if resnet_model_weights is None:
        logging.info("ResNet model weights not provided, training from scratch.")
    else:
        assert os.path.exists(resnet_model_weights), "ResNet model weights not found."
    if pointvector_weights is None:
        logging.info("PointVector-S model weights not provided, training from scratch.")
    else:
        assert os.path.exists(pointvector_weights), "PointVector-S model weights not found."

    # Test if we can load ResNet model weights
    # resnet_model_weights = "/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/2025-01-21-15-02-20_resnet18_epoch_9_acc_79.25.pth"
    model.load_weights(
        resnet_weights_path=resnet_model_weights, pointvector_weights_path=pointvector_weights,
        mlp_weights_path=None, map_location=device
    )
    model.to(device)

    cfg.batch_size = args.batch_size

    # Setup wandb
    assert isinstance(args.use_wandb, bool), "The use_wandb must be a boolean."
    if args.use_wandb:
        cfg.wandb.use_wandb = True
        cfg.wandb.project = cfg.project_name
        wandb.init(project=cfg.wandb.project, name=experiment_name)
        wandb.config.update(args)
        wandb.save(log_file)

    """
    Training
    """
    # Training arguments
    cfg.epochs = args.epochs
    cfg.num_workers = args.num_workers
    if cfg.num_workers == 0:
        logging.warning("The number of workers is set to 0, which may slow down the training process.")

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
    with_class_weights = args.with_class_weights
    if with_class_weights:
        train_labels = train_dataset.df["class_id"].values
        class_weights = calculate_class_weights(train_labels)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    best_val_overall_acc = -1.0
    best_epoch = -1
    cur_best_model_fp = None
    patience = args.epochs  # Early stopping patience (will stop training if the validation accuracy does not improve after this number of epochs)
    epochs_without_improvement = 0
    accumulation_steps = max(1, int(args.grad_accum_steps))

    for epoch in range(1, cfg.epochs + 1):
        train_pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(),
                          desc=f"Train Epoch [{epoch}/{cfg.epochs}]")
        loss_meter = AverageMeter()
        train_cm = ConfusionMatrix(num_classes=cfg.num_classes)

        optimizer.zero_grad()  # ensure grads are zero at the start of each epoch
        model.train()  # make sure we are in train model
        for idx, (fn, data) in train_pbar:

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            points = data['x']
            target = data['y']

            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()

            logits = model(data)
            loss = criterion(logits, target)

            # Dividing the loss by the accumulation steps keeps the scale of the gradients similar to what you would expect from a full batch
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            # step on schedule or at the end of the loader
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # update confusion matrix
            train_cm.update(logits.argmax(dim=1), target)
            loss_meter.update(loss.item())

        # Calculate the accuracy and overall accuracy
        train_loss = loss_meter.avg
        train_macc, train_oacc, accs = train_cm.all_acc()
        if args.backbone_lr > 0:
            backbone_lr = optimizer.param_groups[0]['lr']
            fusion_lr = optimizer.param_groups[2]['lr']
        else:
            backbone_lr = 0
            fusion_lr = optimizer.param_groups[0]['lr']

        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_macc": train_macc,
                "train_oacc": train_oacc,
                "train_cm": train_cm.get_wandb_table(train_dataset.classes),
                "fusion_lr": fusion_lr,
                "backbone_lr": backbone_lr,
                "epoch": epoch
            })

        # Log the training results
        logging.info(
            f"Train: Overall acc (%): {train_oacc:.1f}%, Loss: {train_loss:.3f}, fusion_lr: {round(fusion_lr, 7)}, backbone_lr: {round(backbone_lr, 7)}")
        for class_idx in range(train_cm.num_classes):
            class_total_train = train_cm.actual[class_idx].item()
            class_correct_train = train_cm.tp[class_idx].item()
            class_acc_train = (class_correct_train / class_total_train) * 100 if class_total_train > 0 else 0
            logging.info(
                f"Train: class {train_dataset.classes[class_idx]} (id: {class_idx}) correct: {class_correct_train}/{class_total_train} ({class_acc_train:.1f}%)")

        """
        VALIDATION
        """
        val_cm = ConfusionMatrix(num_classes=cfg.num_classes)
        is_best = False

        with torch.no_grad():
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
                points = points[:, :cfg.num_points]  # TODO this is potentially problematic if points are sorted
                data['pos'] = points[:, :, :3].contiguous()
                data['x'] = points[:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()

                logits = model(data)
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
                logging.info(
                    f"Val: class {val_dataset.classes[class_idx]} "
                    f"(id: {class_idx}) correct: {class_correct_val}/{class_total_val} ({class_acc_val:.1f}%)"
                )

            # check if the current model is the best model
            is_best = val_overall_acc > best_val_overall_acc

            if is_best:
                epochs_without_improvement = 0
                best_val_overall_acc = val_overall_acc
                best_epoch = epoch
                logging.info(f"Best model found at epoch {epoch}, saving model...")

                # Delete the previous best model (*.pth file)
                prev_best_model = glob.glob1(cfg.experiment_dir, "multi_modal_fusion_model_*.pth")
                if len(prev_best_model) > 0:
                    logging.info(f"Deleting previous best model: {prev_best_model[0]}")
                    os.remove(os.path.join(cfg.experiment_dir, prev_best_model[0]))

                logging.info(f"Saving the best model with overall accuracy: {best_val_overall_acc:.2f}%")
                cur_best_model_fp = os.path.join(
                    cfg.experiment_dir, f"multi_modal_fusion_model_{best_val_overall_acc:.2f}_epoch_{epoch}.pth"
                )
                torch.save(model.state_dict(), cur_best_model_fp)

                # Write the results to a csv file
                pred_label_fp = os.path.join(
                    cfg.experiment_dir, f"val_prediction_labels_epoch_{epoch}_oa_{round(best_val_overall_acc, 1)}.csv"
                )
                with open(pred_label_fp, "w") as f:
                    f.write("image_path,prediction,label,correct,confidence\n")
                    for img_path, pred, label, conf in zip(
                            val_file_path_list, val_pred_list, val_label_list, val_conf_list
                    ):
                        f.write(
                            f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},"
                            f"{round(conf * 100, 0)}\n"
                        )
                    # Write overall high, low and total accuracy
                    low_total = val_cm.actual[0].item()
                    low_correct = val_cm.tp[0].item()
                    low_acc = (low_correct / low_total) * 100 if low_total > 0 else 0
                    f.write(f"Low bio correct,{low_correct},{low_total},{low_acc}\n")
                    high_total = val_cm.actual[1].item()
                    high_correct = val_cm.tp[1].item()
                    high_acc = (high_correct / high_total) * 100 if high_total > 0 else 0
                    f.write(f"High bio correct,{high_correct},{high_total},{high_acc}\n")
                    f.write(
                        f"Overall validation accuracy,"
                        f"{val_cm.tp.sum().item()},{val_cm.actual.sum().item()},{best_val_overall_acc}\n"
                    )
                    f.write(f"Mean validation accuracy,,,{val_macc}\n")
                f.close()

                wandb.log({
                    "best_val_oacc": best_val_overall_acc,
                    "best_val_cm": val_cm.get_wandb_table(train_dataset.classes),
                    "epoch": epoch
                })
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logging.info(f"Early stopping after {patience} epochs without improvement.")
                    break

            if cfg.wandb.use_wandb:
                wandb.save(pred_label_fp)
                wandb.log({
                    "val_acc": val_macc,
                    "val_oacc": val_overall_acc,
                    "val_cm": val_cm.get_wandb_table(train_dataset.classes),
                    "epoch": epoch
                })

        scheduler.step(epoch)

    test_dataset = BioVista2D3D(
        data_root=args.source, split='test', transform=transform,
        orthophoto_channels=args.orthophoto_channels, in_memory=False, seed=cfg.seed
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # test_loader.dataset.df = test_loader.dataset.df.sample(200, random_state=cfg.seed)
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
    with torch.no_grad():
        test_cm = ConfusionMatrix(num_classes=cfg.num_classes)
        for i, (fn, data) in tqdm(enumerate(test_loader), total=test_loader.__len__(), desc=f"Testing:"):

            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)

            labels = data['y']

            data['pos'] = data['x'][:, :, :3].contiguous()
            data['x'] = data['x'][:, :, :4].transpose(1, 2).contiguous()

            # # Forward pass
            # _2D_features = model.forward_2D_feature_encodings(data['img'])
            # _3D_features = model.forward_3D_feature_encodings(data)
            # features_2D_3D = torch.cat([_2D_features, _3D_features], dim=1)

            # Save the 2D and 3D encodings
            # image_file_name = os.path.basename(fn[0]) + "_30m.png"
            # _2D_feature_dir = os.path.join(cfg.experiment_dir, "resnet_encodings")
            # if not os.path.exists(_2D_feature_dir):
            #     os.makedirs(_2D_feature_dir, exist_ok=True)
            # _2D_feature_fp = os.path.join(_2D_feature_dir, image_file_name.replace(".png", ".npy"))

            # if not os.path.exists(_2D_feature_fp):
            #     np.save(_2D_feature_fp, _2D_features.cpu().numpy())

            # point_cloud_file_name = os.path.basename(fn[0]) + "_30m.npz"
            # _3D_feature_dir = os.path.join(cfg.experiment_dir, "pointvector_encodings")
            # if not os.path.exists(_3D_feature_dir):
            #     os.makedirs(_3D_feature_dir, exist_ok=True)
            # _3D_feature_fp = os.path.join(_3D_feature_dir, point_cloud_file_name.replace(".npz", ".npy"))

            # if not os.path.exists(_3D_feature_fp):
            #     np.save(_3D_feature_fp, _3D_features.cpu().numpy())

            logits = model(data)

            test_cm.update(logits.argmax(dim=1), labels)
            _, preds = torch.max(logits, 1)
            # Calculate the confidence scores between 0-100% for the predictions
            confidences = torch.nn.functional.softmax(logits, dim=1)
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
            f.write(f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf * 100, 0)}\n")
        # Write overall high, low and total accuracy
        f.write(f"Low bio correct,{low_correct_test.item()},{n_low_bio_samples_test.item()},{overall_val_acc_low}\n")
        f.write(
            f"High bio correct,{high_correct_test.item()},{n_high_bio_samples_test.item()},{overall_val_acc_high}\n"
        )
        f.write(
            f"Overall test accuracy,"
            f"{low_correct_test.item() + high_correct_test.item()},{len(test_dataset)},{round(overall_test_acc, 2)}\n"
        )
        f.write(f"Mean test accuracy,,,{round((overall_val_acc_low + overall_val_acc_high) / 2, 2)}\n")
    f.close()

    if args.use_wandb:
        wandb.log({
            "test_macc": round((overall_val_acc_low + overall_val_acc_high) / 2, 2),
            "test_oacc": overall_test_acc,
            "test_low_bio_acc": overall_val_acc_low,
            "test_high_bio_acc": overall_val_acc_high,
            "test_cm": test_cm.get_wandb_table(train_dataset.classes),
        })
