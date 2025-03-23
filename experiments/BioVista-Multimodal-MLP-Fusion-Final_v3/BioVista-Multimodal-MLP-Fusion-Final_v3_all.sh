
# Experiment 1 - ResNet 1 and PointVector 1 Active Backbones
python Train_MultiModalFusionModel.py --project_name BioVista-Multimodal-MLP-Fusion-Final_v3 --num_workers 8 --epochs 20 --batch_size 8 --fusion_lr 0.0001 --backbone_lr 0 
--pointvector_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/3D-ALS-point-cloud-PointVector/BioVista-Query-Ball-Radius-and-Scaling_v1/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth
--resnet_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/2D-Orthophotos-ResNet/BioVista-ResNet-18-vs-34-vs-50_v1/2025-01-22-22-23-18_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-22-23-18_resnet18_epoch_13_acc_79.9.pth

# Experiment 2 - ResNet 1 and PointVector 1 Frozen backbones
python Train_MultiModalFusionModel.py --project_name BioVista-Multimodal-MLP-Fusion-Final_v3 --num_workers 8 --epochs 20 --batch_size 8 --fusion_lr 0.0001 --backbone_lr 0.000001 
--pointvector_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/3D-ALS-point-cloud-PointVector/BioVista-Query-Ball-Radius-and-Scaling_v1/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-04-00-36-38_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth
--resnet_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/2D-Orthophotos-ResNet/BioVista-ResNet-18-vs-34-vs-50_v1/2025-01-22-22-23-18_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-22-23-18_resnet18_epoch_13_acc_79.9.pth

# Experiment 3 - ResNet 2 and PointVector 2 Active Backbones
python Train_MultiModalFusionModel.py --project_name BioVista-Multimodal-MLP-Fusion-Final_v3 --num_workers 8 --epochs 20 --batch_size 8 --fusion_lr 0.0001 --backbone_lr 0 
--pointvector_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/3D-ALS-point-cloud-PointVector/BioVista-Data-Augmentation_v2/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth
--resnet_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/2D-Orthophotos-ResNet/BioVista-ResNet-18-RGBNIR-Channels_v1/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/2025-01-21-15-02-20_resnet18_epoch_9_acc_79.25.pth

# Experiment 4 - ResNet 2 and PointVector 2 Frozen backbones
python Train_MultiModalFusionModel.py --project_name BioVista-Multimodal-MLP-Fusion-Final_v3 --num_workers 8 --epochs 20 --batch_size 8 --fusion_lr 0.0001 --backbone_lr 0.000001 
--pointvector_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/3D-ALS-point-cloud-PointVector/BioVista-Data-Augmentation_v2/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-05-12-42-43_BioVista-Data-Augmentation_v2_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth
--resnet_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/2D-Orthophotos-ResNet/BioVista-ResNet-18-RGBNIR-Channels_v1/2025-01-21-15-02-20_BioVista-ResNet-18-RGBNIR-Channels_v1_resnet18_channels_NGB/2025-01-21-15-02-20_resnet18_epoch_9_acc_79.25.pth

# Experiment 5 - ResNet 3 and PointVector 3 Active Backbones
python Train_MultiModalFusionModel.py --project_name BioVista-Multimodal-MLP-Fusion-Final_v3 --num_workers 8 --epochs 20 --batch_size 8 --fusion_lr 0.0001 --backbone_lr 0 
--pointvector_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/3D-ALS-point-cloud-PointVector/BioVista-Query-Ball-Radius-and-Scaling_v1/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth
--resnet_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/2D-Orthophotos-ResNet/BioVista-ResNet-18-vs-34-vs-50_v1/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth

# Experiment 6 - ResNet 3 and PointVector 3 Frozen backbones
python Train_MultiModalFusionModel.py --project_name BioVista-Multimodal-MLP-Fusion-Final_v3 --num_workers 8 --epochs 20 --batch_size 8 --fusion_lr 0.0001 --backbone_lr 0.000001 
--pointvector_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/3D-ALS-point-cloud-PointVector/BioVista-Query-Ball-Radius-and-Scaling_v1/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5/checkpoint/2025-02-03-15-27-21_BioVista-Query-Ball-Radius-and-Scaling-v1_pointvector-s_channels_xyzh_npts_16384_qb_r_0.65_qb_s_1.5_ckpt_best.pth
--resnet_weights /home/create.aau.dk/fd78da/datasets/BioVista/Forest-Biodiversity-Potential/experiments/2D-Orthophotos-ResNet/BioVista-ResNet-18-vs-34-vs-50_v1/2025-01-22-21-35-49_BioVista-ResNet-18-vs-34-vs-50_v1_resnet18_channels_NGB/2025-01-22-21-35-49_resnet18_epoch_15_acc_78.67.pth