docker run --runtime=nvidia --gpus all --ipc=host -v /home/simon/aau/BioVista-PointNext/:/workspace/src -v /media/simon/Elements/BioVista/datasets/test_3D_point_cloud_pipeline/:/workspace/dataset --name biovista-pointnext-container -it biovistapointnext:latest