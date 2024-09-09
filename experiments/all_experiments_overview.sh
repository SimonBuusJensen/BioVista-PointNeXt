
# python train_classifier.py --num_points 8192 --qb_radius 0.15 --batch_size_train 4 --cfg /workspace/src/cfgs/biovista/pointvector-s.yaml

# Batch 1
python train_classifier.py --num_points 8192 --qb_radius 0.7 --batch_size_train 4 --cfg cfgs/biovista/pointvector-s.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv
python train_classifier.py --num_points 16384 --qb_radius 0.15 --batch_size_train 2 --cfg cfgs/biovista/pointvector-s.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv

# Batch 2
python train_classifier.py --num_points 16384 --qb_radius 0.7 --batch_size_train 2 --cfg cfgs/biovista/pointvector-s.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv

# Batch 3 PointVector-XL
python train_classifier.py --num_points 8192 --qb_radius 0.15 --batch_size_train 4 --cfg cfgs/biovista/pointvector-xl.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv
python train_classifier.py --num_points 16384 --qb_radius 0.15 --batch_size_train 2 --cfg cfgs/biovista/pointvector-xl.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv

# Batch 4 PointVector-XL
python train_classifier.py --num_points 8192 --qb_radius 0.7 --batch_size_train 4 --cfg cfgs/biovista/pointvector-xl.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv
python train_classifier.py --num_points 16384 --qb_radius 0.7 --batch_size_train 2 --cfg cfgs/biovista/pointvector-xl.yaml --dataset_csv /home/create.aau.dk/fd78da/datasets/BioVista/datasets/100_high_and_100_low_HNV-forest-proxy-samples/100_high_and_100_low_HNV-forest-proxy-samples_30_m_circles_dataset.csv
