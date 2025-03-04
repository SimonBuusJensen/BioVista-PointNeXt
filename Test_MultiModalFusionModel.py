import argparse
import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from openpoints.utils import EasyConfig, cal_model_parm_nums, set_random_seed
from openpoints.models.backbone.pointvector import PointVectorEncoder
from openpoints.models.classification.cls_base import ClsHead
from openpoints.dataset import BioVista2D3D
from fusion_classifier.FeatureDataset import FeatureDataset


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()

        self.resnet = resnet18(pretrained=False)

        # Adjust the input channels of the conv1 layer to accomodate for 1, 3 or 4 channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            num_ftrs, num_classes)  # Adjust output layer

    def forward(self, x):
        x = self.resnet(x)
        return x

    def get_feature_encodings(self, x):
        """
        methods for extracting encodings from the model 
        """
        with torch.no_grad():
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            x = self.resnet.avgpool(x)
            features = torch.flatten(x, 1)
            return features


class MLPModel(nn.Module):
    def __init__(self, input_size=1024, output_size=2, option=3, dropout_rate=0.0):
        super(MLPModel, self).__init__()

        if option == 1:
            hidden_sizes = [512]  # Simple
        elif option == 2:
            hidden_sizes = [512, 256]  # Deeper
        elif option == 3:
            hidden_sizes = [1024, 512, 256]  # Most expressive
        else:
            raise ValueError("Invalid option. Choose 1, 2, or 3.")

        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            # Normalization for stability
            layers.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))  # Prevent overfitting

            in_features = hidden_size  # Set input size for next layer

        # Final output layer
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiModalFusionModel(nn.Module):
    def __init__(self,
                 num_classes=2,
                 fusion_input_size=1024):
        super(MultiModalFusionModel, self).__init__()
        # Instantiate image backbone.
        # Note: Here we use get_feature_encodings to get a feature vector.
        # You might need to adjust the final feature dimension.
        self.image_backbone = ResNetClassifier(num_classes=2,
                                               in_channels=3)

        # Instantiate point cloud backbone.
        # Use the same configuration you use for PointVector-S. Adjust parameters as needed.
        self.point_backbone = nn.Module()
        self.point_backbone.encoder = PointVectorEncoder(
            in_channels=4,
            width=32,
            blocks=[1, 1, 1, 1, 1, 1],
            strides=[1, 2, 2, 2, 2, 1],
            sa_layers=2,
            sa_use_res=True,
            radius=0.65,
            radius_scaling=1.5,
            nsample=32,
            expansion=4,
            flag=0,
            aggr_args={'feature_type': 'dp_fj', 'reduction': 'max'},
            group_args=EasyConfig({'NAME': 'ballquery', 'normalize_dp': True}),
            conv_args={'order': 'conv-norm-act'},
            act_args={'act': 'leakyrelu'},
            norm_args={'norm': 'bn'}
        )

        self.point_backbone.prediction = ClsHead(num_classes=num_classes,
                                               in_channels=512,
                                               mlps=[512, 256],
                                               norm_args={'norm': 'bn1d'},
                                               )
        
        # Fusion head: expects concatenated features.
        # For example, if both backbones output 512-d features, then 512 + 512 = 1024.
        self.fusion_head = MLPModel(input_size=fusion_input_size,
                                    output_size=num_classes,
                                    option=3,
                                    dropout_rate=0.0)

    def forward(self, data):
        # Extract image features. (Assume image is (B, C, H, W))
        image_features = self.image_backbone.get_feature_encodings(data['img'])

        # Extract point cloud features.
        # Here we use forward_cls_feat; ensure your point cloud data is in the expected format.
        point_features = self.point_backbone.encoder.forward_cls_feat(data)

        # Concatenate features along the feature dimension.
        fused_features = torch.cat([image_features, point_features], dim=1)

        # Pass the fused vector through the MLP fusion head.
        out = self.fusion_head(fused_features)
        return out
    
    def forward_MLP_predictions(self, features_2D_3D):
        # Forward pass through the MLP model
        outputs = self.fusion_head(features_2D_3D)
        return outputs

    def forward_2D_feature_encodings(self, image):
        # Extract Features from the input image using the ResNet-18 backbone.
        image_features = self.image_backbone.get_feature_encodings(image)
        return image_features

    def forward_3D_feature_encodings(self, point_cloud):
        # Extract Features from the 3D point cloud using the PointVector-S backbone.
        point_features = self.point_backbone.encoder.forward_cls_feat(point_cloud)
        return point_features

    def forward_2D_predictions(self, image):
        # Extract Features from the input image using the ResNet-18 backbone.
        prediction = self.image_backbone(image)
        return prediction

    def forward_3D_predictions(self, point_cloud_data):
        # Extract Features from the 3D point cloud using the PointVector-S backbone.
        point_features = self.point_backbone.encoder.forward_cls_feat(point_cloud_data)
        prediction = self.point_backbone.prediction(point_features)
        return prediction

    def load_weights(self,
                     resnet_weights_path=None,
                     pointvector_weights_path=None,
                     mlp_weights_path=None,
                     multimodal_weights_path=None,
                     map_location="cpu"):
        """
        Load weights from separate files or a full model checkpoint.

        Args:
            resnet_weights_path (str): Path to a checkpoint for the ResNet-18 backbone.
            pointvector_weights_path (str): Path to a checkpoint for the PointVector-S backbone.
            mlp_weights_path (str): Path to a checkpoint for the MLP fusion head.
            multimodal_weights_path (str): Path to a checkpoint for the entire MultiModalFusionModel.
            map_location (str): Device mapping for torch.load.
        """
        if multimodal_weights_path:
            # Load a full model checkpoint
            state_dict = torch.load(multimodal_weights_path, map_location=map_location)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded full MultiModalFusionModel weights.")
            return

        if resnet_weights_path:
            state_dict = torch.load(
                resnet_weights_path, map_location=map_location)
            # Optionally adjust keys if necessary.
            # Directly load the weights into the ResNet model of the image backbone.
            self.image_backbone.load_state_dict(state_dict)
            print("Loaded ResNet weights.")
        
        if pointvector_weights_path:
            if not os.path.exists(pointvector_weights_path):
                raise NotImplementedError('no checkpoint file from path %s...' % pointvector_weights_path)
            # load state dict
            state_dict = torch.load(pointvector_weights_path, map_location=map_location)

            # parameter resume of base model
            ckpt_state_dict = state_dict['model']
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt_state_dict.items()}
        
            self.point_backbone.load_state_dict(base_ckpt)
            # epoch = state_dict.get('epoch', -1)
            print("Loaded PointVector-S weights.")

        if mlp_weights_path:
            state_dict = torch.load(mlp_weights_path, map_location=map_location)
            # Optionally adjust keys if necessary.
            # Directly load the weights into the MLP model of the fusion head.
            self.fusion_head.load_state_dict(state_dict)
            print("Loaded MLP weights.")
        


def load_checkpoint(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)
    # load state dict
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # parameter resume of base model
    ckpt_state_dict = state_dict['model']
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt_state_dict.items()}
  
    model.load_state_dict(base_ckpt)
    epoch = state_dict.get('epoch', -1)
    
    return epoch

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
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth")
                        default=None)
    parser.add_argument("--features_dir_2d", type=str, help="Path to a directory containing the 2D features of the images.",
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/resnet_encodings/")
                        default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/BioVista-Multimodal-Fusion-Active-Weights-Test/2025-03-04-14-50-13-BioVista-Multimodal-Fusion-Active-Weights-Test/resnet_encodings")
                        # default=None)

    parser.add_argument('--pointvector_weights', type=str, help='PointVector-S weights file',
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth")
                        default=None)
    parser.add_argument("--features_dir_3d", type=str, help="Path to a directory containing the 3D features of the point clouds.",
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/3D-ALS-point-cloud-PointVector/2025-02-05-21-52-36_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/pointvector_encodings/")
                        default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/BioVista-Multimodal-Fusion-Active-Weights-Test/2025-03-04-14-50-13-BioVista-Multimodal-Fusion-Active-Weights-Test/pointvector_encodings")
                        # default=None)

    parser.add_argument('--mlp_weights', type=str, help='MLP weights file', 
                        # default="/workspace/datasets/experiments/2D-3D-Fusion/MLP-Fusion/Baseline-Frozen/2025-02-20-17-32-55_365_MLP-2D-3D-Fusion_BioVista-MLP-Fusion-Same-Features-v2/mlp_model_81.56_epoch_11.pth")
                        # default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/2025-02-20-17-32-55_365_MLP-2D-3D-Fusion_BioVista-MLP-Fusion-Same-Features-v2/mlp_model_81.56_epoch_11.pth")
                        default=None)
    
    parser.add_argument('--multi_modal_weights', type=str, help='MultiModalFusionModel weights file', 
                        default="/home/simon/data/BioVista/datasets/Forest-Biodiversity-Potential/experiments/2D-3D-Fusion/MLP-Fusion/BioVista-Multimodal-Fusion-Active-Weights-Test/2025-03-04-14-50-13-BioVista-Multimodal-Fusion-Active-Weights-Test/multi_modal_fusion_model_69.00_epoch_2.pth")
    
    
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    
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
    mlp_weights = args.mlp_weights
    multi_modal_weights = args.multi_modal_weights
    features_dir_2d = args.features_dir_2d
    features_dir_3d = args.features_dir_3d
    
    if resnet_model_weights is not None:
        assert os.path.exists(resnet_model_weights), "ResNet model weights not found."
    if pointvector_weights is not None:
        assert os.path.exists(pointvector_weights), "PointVector-S model weights not found."
    if mlp_weights is not None:
        assert os.path.exists(mlp_weights), "MLP model weights not found."  
    if multi_modal_weights is not None:
        assert os.path.exists(multi_modal_weights), "MultiModalFusionModel weights not found."

    # In case of MLP weights, we need either the 2D or 3D features directory or the pre-trained model weights
    if features_dir_2d is not None:
        assert os.path.exists(features_dir_2d), f"2D features directory not found: {features_dir_2d}"
    if features_dir_3d is not None:
        assert os.path.exists(features_dir_3d), f"3D features directory not found: {features_dir_3d}"


    # Test if we can load ResNet model weights
    # resnet_model_weights = "/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/2025-01-21-15-02-20_resnet18_epoch_9_acc_79.25.pth"
    output_dir = os.path.dirname(multi_modal_weights)
    model.load_weights(resnet_weights_path=resnet_model_weights, 
                       pointvector_weights_path=pointvector_weights, 
                       mlp_weights_path=mlp_weights, 
                       multimodal_weights_path=multi_modal_weights,
                       map_location=device)
    model.to(device)

    from torchvision.transforms import Compose
    from openpoints.transforms import PointsToTensor, PointCloudXYZAlign
    transform = Compose([PointsToTensor(), PointCloudXYZAlign(normalize_gravity_dim=False)])
    # test_dataset = BioVista2D3D(data_root=args.source, split='test', transform=transform, seed=cfg.seed)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # test_loader.dataset.df = test_loader.dataset.df.sample(100, random_state=cfg.seed)
    
    test_dataset = FeatureDataset(csv_file=args.source, feature_dir_2d=features_dir_2d, feature_dir_3d=features_dir_3d, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print("Successfully loaded test dataset. with {} samples".format(len(test_dataset)))

    test_acc = 0.0
    high_correct = 0
    low_correct = 0
    n_high_bio_samples = 0
    n_low_bio_samples = 0
    pred_list = []
    conf_list = []
    label_list = []
    file_path_list = []

    model.eval()
    with torch.set_grad_enabled(False):
        for fn, X_test_batch, y_test_batch in tqdm(test_loader, desc="Evaluating on test set", total=test_loader.__len__()):
            
            # Move tensors to the same device
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            # Forward pass

            outputs = model.forward_MLP_predictions(X_test_batch)
            confidences = torch.nn.functional.softmax(outputs, dim=1)
            confidences = torch.max(confidences, 1)[0]

            _, preds = torch.max(outputs, 1)
            labels = torch.max(y_test_batch, 1)[1]
            test_acc += torch.sum(preds == labels.data)

            high_correct += torch.sum((preds == labels) & (labels == 1))
            low_correct += torch.sum((preds == labels) & (labels == 0))

            n_high_bio_samples += torch.sum(labels == 1)
            n_low_bio_samples += torch.sum(labels == 0)

            # Append results
            pred_list.extend(preds.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            file_path_list.extend(fn)
            conf_list.extend(confidences.cpu().detach().numpy())


        # for i, (fn, data) in tqdm(enumerate(test_loader), total=test_loader.__len__()):

        #     for key in data.keys():
        #         data[key] = data[key].cuda(non_blocking=True)

        #     labels = data['y'].to(device)
            
        #     data['pos'] = data['x'][:, :, :3].contiguous()
        #     data['x'] = data['x'][:, :, :4].transpose(1, 2).contiguous()

        #     # Forward pass
        #     _2D_features = model.forward_2D_feature_encodings(data['img'])
        #     _3D_features = model.forward_3D_feature_encodings(data)
        #     features_2D_3D = torch.cat([_2D_features, _3D_features], dim=1)
            
        #     # Save the 2D and 3D encodings
        #     image_file_name = os.path.basename(fn[0]) + "_30m.png"
        #     _2D_feature_dir = os.path.join(os.path.dirname(resnet_model_weights), "resnet_encodings")
        #     if not os.path.exists(_2D_feature_dir):
        #         os.makedirs(_2D_feature_dir, exist_ok=True)
        #     _2D_feature_fp = os.path.join(_2D_feature_dir, image_file_name.replace(".png", ".npy"))
            
        #     if not os.path.exists(_2D_feature_fp):
        #         np.save(_2D_feature_fp, _2D_features.cpu().numpy())
            
        #     point_cloud_file_name = os.path.basename(fn[0]) + "_30m.npz"
        #     _3D_feature_dir = os.path.join(os.path.dirname(os.path.dirname(pointvector_weights)), "pointvector_encodings")
        #     if not os.path.exists(_3D_feature_dir):
        #         os.makedirs(_3D_feature_dir, exist_ok=True)
        #     _3D_feature_fp = os.path.join(_3D_feature_dir, point_cloud_file_name.replace(".npz", ".npy"))
            
        #     if not os.path.exists(_3D_feature_fp):
        #         np.save(_3D_feature_fp, _3D_features.cpu().numpy())
            
        #     outputs = model.forward_MLP_predictions(features_2D_3D)
        #     _, preds = torch.max(outputs, 1)
        #     # Calculate the confidence scores between 0-100% for the predictions
        #     confidences = torch.nn.functional.softmax(outputs, dim=1)
        #     confidences = torch.max(confidences, 1)[0]

        #     test_acc += torch.sum(preds == labels.data)
        #     high_correct += torch.sum((preds == labels.data) & (labels == 1))
        #     low_correct += torch.sum((preds == labels.data) & (labels == 0))

        #     n_high_bio_samples += torch.sum(labels == 1)
        #     n_low_bio_samples += torch.sum(labels == 0)

        #     # Append the predictions and labels to the lists
        #     pred_list.extend(preds.cpu().numpy())
        #     label_list.extend(labels.cpu().numpy())
        #     file_path_list.extend(fn)
        #     # Append the confidence scores as float with 2 decimals
        #     conf_list.extend(confidences.cpu().detach().numpy())

    # Calculate the overall validation accuracy
    overall_val_acc = round(test_acc.item() / len(test_dataset) * 100, 2)
    if n_high_bio_samples.item() == 0:
        overall_val_acc_high = 0.0
    else:
        overall_val_acc_high = round(high_correct.item() / n_high_bio_samples.item() * 100, 2)

    if n_low_bio_samples.item() == 0:
        overall_val_acc_low = 0.0
    else:
        overall_val_acc_low = round(low_correct.item() / n_low_bio_samples.item() * 100, 2)

    # Write the image_paths, predictions and labels to a csv file
    pred_label_fp = os.path.join(
        output_dir, f"prediction_labels_from_the_mulitmodal_fusion_model.csv")
    with open(pred_label_fp, "w") as f:
        f.write("image_path,prediction,label,correct,confidence\n")
        for img_path, pred, label, conf in zip(file_path_list, pred_list, label_list, conf_list):
            f.write(f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
        # Write overall high, low and total accuracy
        f.write(f"Low bio correct,{low_correct.item()},{n_low_bio_samples.item()},{overall_val_acc_low}\n")
        f.write(f"High bio correct,{high_correct.item()},{n_high_bio_samples.item()},{overall_val_acc_high}\n")
        f.write(f"Overall test accuracy,{low_correct.item() + high_correct.item()},{len(test_dataset)},{overall_val_acc}\n")
        f.write(f"Mean test accuracy,,,{(round(overall_val_acc_low + overall_val_acc_high) / 2, 2)}\n")
    f.close()
