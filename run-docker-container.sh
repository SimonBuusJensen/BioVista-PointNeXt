docker run --runtime=nvidia --gpus all --ipc=host -v /home/simon/aau/BioVista-PointNext/:/workspace/src -v /media/simon/Elements/BioVista/datasets/:/workspace/datasets --name biovista-pointnext-container -it biovistapointnext:latest