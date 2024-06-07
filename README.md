# ground_truth_generation


## Installation

1. execute the script to create python venv environment:
```
cd ~/ros2_ws/ground_truth_generation/
./create_venv.sh
```

## Introduction

1. download the dataset from the following link:
```
mkdir -p ~/dataset/RTS/ && cd ~/dataset/RTS/
wget "dataset_link"
```

2. execute the ground_truth_RTS generation script to compute your data :
```
cd ~/ros2_ws/ground_truth_generation/
./ground_truth_RTS.sh
```
