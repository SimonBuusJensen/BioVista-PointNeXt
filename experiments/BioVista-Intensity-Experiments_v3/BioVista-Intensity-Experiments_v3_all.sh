
# TODO: Missing to add data augmentation 

# 1. Baseline experiments - without intensity
python train_classifier.py --channels xyzh --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzh --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

python train_classifier.py --channels xyzh --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzh --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

# 2. With intensity but no normalization
python train_classifier.py --channels xyzhi --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

python train_classifier.py --channels xyzhi --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

# 3. With intensity and normalization by the mode of ground points
python train_classifier.py --channels xyzhi --with_normalize_intensity True --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --with_normalize_intensity True --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

python train_classifier.py --channels xyzhi --with_normalize_intensity True --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --with_normalize_intensity True --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

# 4. With intensity and normalization by the mode of ground points and scale of 30
python train_classifier.py --channels xyzhi --with_normalize_intensity True --normalize_intensity_scale 30.0 --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --with_normalize_intensity True --normalize_intensity_scale 30.0 --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3

python train_classifier.py --channels xyzhi --with_normalize_intensity True --normalize_intensity_scale 30.0 --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3
python train_classifier.py --channels xyzhi --with_normalize_intensity True --normalize_intensity_scale 30.0 --num_points 16384 --qb_radius 0.65 --qb_radius_scaling 1.5 --batch_size 8 --num_workers 12 --lr 0.0001 --project_name BioVista-Intensity-Experiments_v3


