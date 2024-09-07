
# python train_classifier.py --num_points 8192 --qb_radius 0.15 --batch_size_train 4 --cfg /workspace/src/cfgs/biovista/pointvector-s.yaml

# Batch 1
python train_classifier.py --num_points 8192 --qb_radius 0.7 --batch_size_train 4 --cfg cfgs/biovista/pointvector-s.yaml
python train_classifier.py --num_points 16384 --qb_radius 0.15 --batch_size_train 2 --cfg cfgs/biovista/pointvector-s.yaml

# Batch 2
python train_classifier.py --num_points 16384 --qb_radius 0.7 --batch_size_train 2 --cfg cfgs/biovista/pointvector-s.yaml

# Batch 3 PointVector-XL
python train_classifier.py --num_points 8192 --qb_radius 0.15 --batch_size_train 4 --cfg cfgs/biovista/pointvector-xl.yaml
python train_classifier.py --num_points 16384 --qb_radius 0.15 --batch_size_train 2 --cfg cfgs/biovista/pointvector-xl.yaml

# Batch 4 PointVector-XL
python train_classifier.py --num_points 8192 --qb_radius 0.7 --batch_size_train 4 --cfg cfgs/biovista/pointvector-xl.yaml
python train_classifier.py --num_points 16384 --qb_radius 0.7 --batch_size_train 2 --cfg cfgs/biovista/pointvector-xl.yaml
