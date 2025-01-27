import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
from collections import Counter
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps


def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)
    
def calculate_class_weights(labels):
    # Count the occurrence of each class
    class_counts = Counter(labels)
    total_count = sum(class_counts.values())
    
    # Inverse of frequency
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    
    # Convert to a tensor
    class_weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float)
    return class_weights_tensor


def main(gpu, cfg):
    
    # Setup logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    
    # Setup seed and device
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)
    
    # build dataset
    cfg.dataset.train.num_points = cfg.num_points
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train'
                                             )
    # DEBUG: Sample 5000 random samples. Change the train_loader.dataset.df to a subset of the original dataset
    # train_loader.dataset.df = train_loader.dataset.df.sample(5000) # TODO: Remove this line

    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    cfg.dataset.val.num_points = cfg.num_points
    val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split=cfg.dataset.val.split
                                           )
    # DEBUG: Sample 500 random samples. Change the val_loader.dataset.df to a subset of the original dataset
    # val_loader.dataset.df = val_loader.dataset.df.sample(500) # TODO: Remove this line
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")  
    cfg.dataset.test.num_points = cfg.num_points
    test_loader = build_dataloader_from_cfg(cfg.batch_size,
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split=cfg.dataset.test.split
                                            )
    
    # DEBUG: Sample 500 random samples. Change the test_loader.dataset.df to a subset of the original dataset
    # test_loader.dataset.df = test_loader.dataset.df.sample(500) # TODO: Remove this line
    logging.info(f"length of testing dataset: {len(test_loader.dataset)}")

    # Setup model
    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
        
    if cfg.cls_weighed_loss:
        cfg.model.criterion_args.weight = calculate_class_weights(train_loader.dataset.df['class_id'].values)
        logging.info('Using class weighted loss')
        logging.info(f"Weight: {cfg.model.criterion_args.weight}")
        
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')

    # criterion, optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes

    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points as model input: {cfg.num_points}")
    
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val_oa = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            if cfg.mode == 'val':
                # validation mode
                epoch, best_val_oa = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder_inv':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint_inv(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    """
    TRAINING
    """
    val_macc, val_oa, best_val_oa, best_epoch = 0., 0., 0., 0
    model.zero_grad()

    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        
        train_loss, train_macc, train_oa, _, train_cm = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        if cfg.wandb.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_macc,
                "train_oa": train_oa,
                "lr": lr,
                "epoch": epoch
            })

        logging.info(f"Mean train acc (%): {train_macc:.1f}%, Train loss: {train_loss:.3f}, lr: {lr}")
        for class_idx in range(train_cm.num_classes):
            class_total_train = train_cm.actual[class_idx].item()
            class_correct_train = train_cm.tp[class_idx].item()
            class_acc_train = (class_correct_train / class_total_train) * 100 if class_total_train > 0 else 0
            logging.info(f"Train: class {cfg.classes[class_idx]} (id: {class_idx}) correct: {class_correct_train}/{class_total_train} ({class_acc_train:.1f}%)")
        
        """
        VALIDATION
        """
        is_best = False
        val_loss_meter = AverageMeter()
        if epoch % cfg.val_freq == 0:
            
            with torch.set_grad_enabled(False):
                pred_list = []
                conf_list = []
                label_list = []
                img_path_list = []
                # Run validation
                model.eval()  # set model to eval mode

                # Set no grad for validation
                val_cm = ConfusionMatrix(num_classes=cfg.num_classes)
                pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
                for idx, (fn, data) in pbar:
                    for key in data.keys():
                        data[key] = data[key].cuda(non_blocking=True)
                    target = data['y']
                    points = data['x']
                    points = points[:, :cfg.num_points]
                    data['pos'] = points[:, :, :3].contiguous()
                    data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

                    # Forward pass
                    logits, val_loss = model.get_logits_loss(data, target)

                    val_cm.update(logits.argmax(dim=1), target)
                    val_loss_meter.update(val_loss.item()) # Update validation loss meter

                    # Save the predictions and labels
                    pred_list.extend(logits.argmax(dim=1).cpu().numpy())

                    confidences = torch.nn.functional.softmax(logits, dim=1)
                    confidences = torch.max(confidences, 1)[0]

                    conf_list.extend(confidences.cpu().numpy())
                    label_list.extend(target.cpu().numpy())
                    img_path_list.extend(fn)

                tp, count = val_cm.tp, val_cm.count
                val_macc, val_oa, _ = val_cm.cal_acc(tp, count)

                is_best = val_oa > best_val_oa
                if is_best:
                    best_val_oa = val_oa
                    best_epoch = epoch
                    logging.info(f'Found new best ckpt at epoch: @E{epoch}')
                    save_checkpoint(cfg, model, epoch, optimizer, scheduler, additioanl_dict={'val_oacc': val_oa}, post_fix="ckpt_best")

                    # Write the results to a csv file
                    # Write the image_paths, predictions and labels to a csv file
                    pred_label_fp = os.path.join(cfg.run_dir, f"epoch_{epoch}_oacc_{round(best_val_oa, 2)}_val_pred_labels.csv")
                    with open(pred_label_fp, "w") as f:
                        f.write("image_path,prediction,label,correct,confidence\n")
                        for img_path, pred, label, conf in zip(img_path_list, pred_list, label_list, conf_list):
                            f.write(
                                f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
                        # Write overall high, low and total accuracy
                        low_total = val_cm.actual[0].item()
                        low_correct = val_cm.tp[0].item() 
                        low_acc = (low_correct / low_total) * 100 if low_total > 0 else 0
                        f.write(f"Low bio correct,{low_correct},{low_total},{low_acc}\n")
                        high_total = val_cm.actual[1].item()
                        high_correct = val_cm.tp[1].item()
                        high_acc = (high_correct / high_total) * 100 if high_total > 0 else 0
                        f.write(f"High bio correct,{high_correct},{high_total},{high_acc}\n")
                        f.write(f"Overall validation accuracy,{val_cm.tp.sum().item()},{val_cm.actual.sum().item()},{val_oa}\n")
                        f.write(f"Mean validation accuracy,,,{best_val_oa}")
                    f.close()

                if cfg.wandb.use_wandb:
                    wandb.log({
                        "best_val_oacc": best_val_oa,
                        "macc_when_best": val_macc,
                        "epoch": epoch
                    })

                logging.info(f"Mean val acc (%): {val_macc:.1f}%, Val loss: {val_loss_meter.avg:.4f}, Val OA: {val_oa:.1f}%")
                for class_idx in range(val_cm.num_classes):
                    class_total_val = val_cm.actual[class_idx].item()
                    class_correct_val = val_cm.tp[class_idx].item()
                    class_acc_val = (class_correct_val / class_total_val) * 100 if class_total_val > 0 else 0
                    logging.info(f"Val: class {cfg.classes[class_idx]} (id: {class_idx}) correct: {class_correct_val}/{class_total_val} ({class_acc_val:.1f}%)")
                    if cfg.wandb.use_wandb:
                        wandb.log({
                            f"val_acc_{cfg.classes[class_idx]}": class_acc_val,
                            "epoch": epoch
                        })

        if cfg.wandb.use_wandb:
            wandb.log({
                "val_acc": val_macc,
                "val_loss": val_loss_meter.avg,  # Log validation loss to wandb
                "val_oa": val_oa,
                "val_macc": val_macc,
                "epoch": epoch
            })
     
        if cfg.sched_on_epoch:
            scheduler.step(epoch)

    """
    TESTING
    """
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    
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
            points = points[:, :cfg.num_points]
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
        test_macc, test_oa, test_accs = test_cm.cal_acc(tp, count)
        pred_label_fp = os.path.join(cfg.run_dir, f"test_pred_labels.csv")
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

    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if cfg.wandb.use_wandb:
        wandb.log({
            "test_oa": test_oa,
            "test_macc": test_macc,
            "test_accuracy_high": high_acc,
            "test_accuracy_low": low_acc,
            "epoch": epoch
        })

   
def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc=f"Train Epoch [{epoch}/{cfg.epochs}]")
    num_iter = 0
    for idx, (fn, data) in pbar:
        
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        
        num_iter += 1
        points = data['x']
        target = data['y']
    
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    loss_meter = AverageMeter()
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())

    for idx, (fn, data) in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :cfg.num_points]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits, loss = model.get_logits_loss(data, target)
        
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())  # Update validation loss meter

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    #test
    return macc, overallacc, accs, cm
