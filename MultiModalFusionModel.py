import argparse
import torch
import os
from tqdm import tqdm
from openpoints.utils import EasyConfig, cal_model_parm_nums
from openpoints.models import build_model_from_cfg
from openpoints.models.backbone.pointvector import PointVectorEncoder
from openpoints.dataset import build_dataloader_from_cfg, BioVista2D3D
import torch
from torchvision.models import resnet18
import torch.nn as nn
from torch.utils.data import DataLoader


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        
        self.resnet = resnet18(pretrained=False)

        # Adjust the input channels of the conv1 layer to accomodate for 1, 3 or 4 channels
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # Adjust output layer

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
            layers.append(nn.BatchNorm1d(hidden_size))  # Normalization for stability
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))  # Prevent overfitting
            
            in_features = hidden_size  # Set input size for next layer
        
        layers.append(nn.Linear(in_features, output_size))  # Final output layer
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
        self.point_backbone = PointVectorEncoder(
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
        
        # Fusion head: expects concatenated features.
        # For example, if both backbones output 512-d features, then 512 + 512 = 1024.
        self.fusion_head = MLPModel(input_size=fusion_input_size,
                                    output_size=num_classes,
                                    option=3,
                                    dropout_rate=0.0)
    
    def forward(self, image, point_cloud):
        # Extract image features. (Assume image is (B, C, H, W))
        image_features = self.image_backbone.get_feature_encodings(image)
        
        # Extract point cloud features. 
        # Here we use forward_cls_feat; ensure your point cloud data is in the expected format.
        point_features = self.point_backbone.forward_cls_feat(point_cloud)
        
        # Concatenate features along the feature dimension.
        fused_features = torch.cat([image_features, point_features], dim=1)
        
        # Pass the fused vector through the MLP fusion head.
        out = self.fusion_head(fused_features)
        return out
    
    def forward_2D_feature_encodings(self, image):
        # Extract Features from the input image using the ResNet-18 backbone.
        image_features = self.image_backbone.get_feature_encodings(image)
        return image_features
    
    def forward_3D_feature_encodings(self, point_cloud):
        # Extract Features from the 3D point cloud using the PointVector-S backbone.
        point_features = self.point_backbone.forward_cls_feat(point_cloud)
        return point_features

    def forward_2D_predictions(self, image):
        # Extract Features from the input image using the ResNet-18 backbone.
        image_features = self.image_backbone(image)
        return image_features
    
    def forward_3D_predictions(self, point_cloud):
        # Extract Features from the 3D point cloud using the PointVector-S backbone.
        point_features = self.point_backbone.forward_cls_feat(point_cloud)
        return point_features
    
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
            state_dict = torch.load(resnet_weights_path, map_location=map_location)
            # Optionally adjust keys if necessary.
            # Directly load the weights into the ResNet model of the image backbone.
            self.image_backbone.resnet.load_state_dict(state_dict, strict=False)
            print("Loaded ResNet weights.")
                          
        return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, help='config file',
                        default="/workspace/src/cfgs/biovista/pointvector-s.yaml")
                        # default="cfgs/biovista/pointvector-s.yaml")
    

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    
    cfg.model.encoder_args.in_channels = 4 # xyzh
    cfg.model.encoder_args.radius = 0.65
    cfg.model.encoder_args.radius_scaling = 1.5 

    # Check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = build_model_from_cfg(cfg.model).to(device)
    model = MultiModalFusionModel()
    model_size = cal_model_parm_nums(model)
    # print(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # Test if we can load ResNet model weights
    resnet_model_weights = "/workspace/datasets/experiments/2D-3D-Fusion/2D-Orthophotos-ResNet/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/2025-01-21-15-02-20_resnet18_epoch_9_acc_79.25.pth"
    output_dir = os.path.dirname(resnet_model_weights)
    model.load_weights(resnet_weights_path=resnet_model_weights)
    model.to(device)

    from torchvision.transforms import Compose
    from openpoints.transforms import PointsToTensor, PointCloudXYZAlign
    transform = Compose([PointsToTensor(), PointCloudXYZAlign()])
    test_dataset = BioVista2D3D(data_root='/workspace/datasets/samples.csv', split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
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
    torch.backends.cudnn.enabled = True
    with torch.set_grad_enabled(False):
        for i, (fn, data) in tqdm(enumerate(test_loader), total=test_loader.__len__()):
            
            for key in data.keys():
                data[key] = data[key].cuda(non_blocking=True)
            
            image = data['img'].to(device)
            points = data['x']
            
            labels = data['y']
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :4].transpose(1, 2).contiguous()
            
            # Forward pass
            outputs = model.forward_2D_predictions(image)

            _, preds = torch.max(outputs, 1)
            # Calculate the confidence scores between 0-100% for the predictions
            confidences = torch.nn.functional.softmax(outputs, dim=1)
            confidences = torch.max(confidences, 1)[0]
            
            test_acc += torch.sum(preds == labels.data)

            high_correct += torch.sum((preds ==
                                        labels.data) & (labels == 1))
            low_correct += torch.sum((preds ==
                                        labels.data) & (labels == 0))

            n_high_bio_samples += torch.sum(labels == 1)
            n_low_bio_samples += torch.sum(labels == 0)

            # Append the predictions and labels to the lists
            pred_list.extend(preds.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            file_path_list.extend(fn)
            # Append the confidence scores as float with 2 decimals
            conf_list.extend(confidences.cpu().detach().numpy())

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
    pred_label_fp = os.path.join(output_dir, f"prediction_labels_from_the_mulitmodal_fusion_model.csv")
    with open(pred_label_fp, "w") as f:
        f.write("image_path,prediction,label,correct,confidence\n")
        for img_path, pred, label, conf in zip(file_path_list, pred_list, label_list, conf_list):
            f.write(
                f"{os.path.basename(img_path)},{pred},{label},{int(pred == label)},{round(conf*100, 0)}\n")
        # Write overall high, low and total accuracy
        f.write(f"Low bio correct,{low_correct.item()},{n_low_bio_samples.item()},{overall_val_acc_low}\n")
        f.write(f"High bio correct,{high_correct.item()},{n_high_bio_samples.item()},{overall_val_acc_high}\n")
        f.write(f"Overall test accuracy,{test_acc.item()},{len(test_dataset)},{overall_val_acc}\n")
        f.write(f"Mean test accuracy,,,{(overall_val_acc_low + overall_val_acc_high) / 2}\n")
    f.close()
            

   