import argparse
import torch
from openpoints.utils import EasyConfig, cal_model_parm_nums
from openpoints.models import build_model_from_cfg
from openpoints.models.backbone.pointvector import PointVectorEncoder
import torch
from torchvision.models import resnet18
import torch.nn as nn


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
    print(model)
    print('Number of params: %.4f M' % (model_size / 1e6))



    """
    BaseCls
        PointVectorEncoder
        SetAbstractioncls (Conv2D 1x1)
        4 x SetAbstractioncls (VPSA)
        SetAbstractioncls (Conv2D 512x512)
    ClsHead
        Linear layers 512 -> 512 -> 256 -> 2
    """