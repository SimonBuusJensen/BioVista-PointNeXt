docker run --runtime=nvidia --gpus all --ipc=host -v /home/simon/aau/BioVista-PointNeXt/:/workspace/src -v /home/simon/data/BioVista/Forest-Biodiversity-Potential/:/workspace/datasets --name biovista-pointnext-container-local-data -it biovistapointnext:latest