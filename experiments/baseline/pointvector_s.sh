

python train_classifier.py --with_class_weights True --channels xyz  --num_points 8192 --qb_radius 0.7 --batch_size 8 --lr 0.0001 --project_name BioVista-Hyperparameter-Search
python train_classifier.py --with_class_weights False --channels xyz  --num_points 8192 --qb_radius 0.7 --batch_size 8 --lr 0.0001 --project_name BioVista-Hyperparameter-Search


# Batch size: 4, 8, 16,
# Number of points: 4096, 8192, 16384
# Learning rate: 0.001, 0.0001, 0.00001
# qb_radius 0.5, 0.7, 0.9
# cls_weighed_loss: True, False
# Number of experiments: 3 * 3 * 3 * 3 * 2 = 162
