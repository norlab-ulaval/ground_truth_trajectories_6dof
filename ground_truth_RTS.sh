#! /bin/bash

DATA=~/dataset/RTS/20220224
mkdir -p $DATA/output/
# python3 RTS_data_reading.py -p $DATA -s -v
python3 RTS_ground_truth_generation.py -p $DATA -d --debug
