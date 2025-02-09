
"""
Examples rows:
Name	project_name	best_val_oacc	test_oa	lr	batch_size	num_points	qb_radius	qb_radius_scaling	channels	with_class_weights	with_point_cloud_jitter	with_point_cloud_rotations	with_point_cloud_scaling	with_normalize_gravity_dim
2025-01-28-01-57-28_BioVista-Hyperparameter-Search_pointvector-s_3304	BioVista-Hyperparameter-Search	77.2018356323242	76.1135940551758	0.0001	8	16384	0.7	1.5	xyz	true	false	false	false	true
2025-01-27-12-22-41_BioVista-Hyperparameter-Search_pointvector-s_1016	BioVista-Hyperparameter-Search	75.9403610229492	75.1173706054688	0.0001	8	8192	0.7	1.5	xyz	false	false	false	false	true
2025-01-28-23-58-13_BioVista-Hyperparameter-Search_pointvector-s_8436	BioVista-Hyperparameter-Search	77.1100921630859	75.0372161865234	0.0001	8	16384	0.7	1.5	xyz	false	true	true	true	true
2025-01-27-18-45-46_BioVista-Hyperparameter-Search_pointvector-s_1833	BioVista-Hyperparameter-Search	75.3899078369141	74.9112548828125	0.0001	8	8192	0.7	1.5	xyz	true	false	false	false	true
2025-01-27-12-22-36_BioVista-Hyperparameter-Search_pointvector-s_879	BioVista-Hyperparameter-Search	76.4220199584961	74.7623977661133	0.0001	8	16384	0.7	1.5	xyz	false	false	false	false	true

"""

"""
Script fro finding all experiments where:
channels == 'xyz'
num_points == 8192
with_class_weights == true
with_point_cloud_jitter == true
with_point_cloud_rotations == true
with_point_cloud_scaling == true
with_normalize_gravity_dim == true
learning_rate == 0.0001
batch_size == 8
"""

import pandas as pd
import os


csv_file = "/home/simon/Downloads/all-pointvector-als-experiments.csv"
# csv_file = "/home/simon/Downloads/wandb_export_2025-02-06T12_14_13.495+01_00.csv"
df = pd.read_csv(csv_file)

# Filter the dataframe
filtered_df = df[
    # (df['project_name'] == 'BioVista-Hyperparameter-Search-v2') &
    (df['channels'] == 'xyzh') &
    # (df['num_points'] == 16384) &
    (df['num_points'] == 24576) &
    (df['qb_radius'] == 0.65) &
    (df['qb_radius_scaling'] == 1.5) &
    # (df['with_class_weights'] == False) &
    # (df['with_point_cloud_jitter'] == False) &
    # (df['with_point_cloud_rotations'] == True) &
    # (df['with_point_cloud_scaling'] == True) &
    # (df['with_normalize_gravity_dim'] == False) &
    (df['lr'] == 0.0001)  & 
    (df['batch_size'] == 8)
]

# Print the filtered dataframe
print(filtered_df)

# Print the min, max, mean and std of the test_oacc
print("Test OACC")
print(round(filtered_df['test_oa'].mean(), 2))
print(round(filtered_df['test_oa'].std(), 2))
print(round(filtered_df['test_oa'].min(), 2))
print(round(filtered_df['test_oa'].max(), 2))

print("Test Acc High")
print(round(filtered_df['test_accuracy_high'].mean(), 2))
print(round(filtered_df['test_accuracy_high'].std(), 2))
print(round(filtered_df['test_accuracy_high'].min(), 2))
print(round(filtered_df['test_accuracy_high'].max(), 2))

print("Test Acc Low")
print(round(filtered_df['test_accuracy_low'].mean(), 2))
print(round(filtered_df['test_accuracy_low'].std(), 2))
print(round(filtered_df['test_accuracy_low'].min(), 2))
print(round(filtered_df['test_accuracy_low'].max(), 2))

print("Test MAcc")
print(round(filtered_df['test_macc'].mean(), 2))
print(round(filtered_df['test_macc'].std(), 2))
print(round(filtered_df['test_macc'].min(), 2))
print(round(filtered_df['test_macc'].max(), 2))
