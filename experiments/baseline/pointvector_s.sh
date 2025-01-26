

python train_classifier.py --num_points 8192 --qb_radius 0.7 --batch_size_train 8 --lr 0.001 --cfg cfgs/biovista/pointvector-s.yaml
python train_classifier.py --num_points 16384 --qb_radius 0.7 --batch_size_train 8 --lr 0.001 --cfg cfgs/biovista/pointvector-s.yaml

python train_classifier.py --num_points 8192 --qb_radius 0.7 --batch_size_train 8 --lr 0.0001 --cfg cfgs/biovista/pointvector-s.yaml
python train_classifier.py --num_points 16384 --qb_radius 0.7 --batch_size_train 8 --lr 0.0001 --cfg cfgs/biovista/pointvector-s.yaml


# Batch size: 4, 8, 16,
# Number of points: 4096, 8192, 16384
# Learning rate: 0.001, 0.0001, 0.00001
# qb_radius 0.5, 0.7, 0.9
# cls_weighed_loss: True, False
# Number of experiments: 3 * 3 * 3 * 3 * 2 = 162
