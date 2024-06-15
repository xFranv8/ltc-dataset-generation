#!/bin/bash
/home/nvidia/Documents/ltc-dataset-generation/simulator/CarlaUE4.sh -world-port=2000 -RenderOffScreen -carla-rpc-port=1212 &
source ~/anaconda3/bin/activate liquidnns
cd /home/nvidia/Documents/ltc-dataset-generation
python main.py
conda deactivate
