

# 3. With intensity and normalization by the mode of ground points
python train_classifier.py --channels xyzhi --with_normalize_intensity True --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --with_normalize_intensity True --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3


