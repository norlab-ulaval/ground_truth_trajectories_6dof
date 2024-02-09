import sys
import os
import argparse
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from tqdm import tqdm
import pandas as pd
from pathlib import Path

################### PARAMETERS #####################
BASE_PATH = Path(__file__).absolute().parents[1]
####################################################

def main(bag_name, output_name, topic_name):

    input_filepath = bag_name

    save_path = Path(BASE_PATH / 'trajectories')
                     
    if not os.path.isdir(input_filepath):
        print('Error: Cannot locate input bag file [%s]' % input_filepath, file=sys.stderr)
        sys.exit(2)
        
    with Reader(input_filepath) as reader:
        if topic_name not in reader.topics:
            print(f'Error: Cannot find topic {topic_name} in bag file {input_filepath}', file=sys.stderr)
        traj = {'timestamp': [], 'x': [], 'y': [], 'z': [], 'q_x': [], 'q_y': [], 'q_z': [], 'q_w': []}
        connections = [x for x in reader.connections if x.topic == topic_name]
        for conn, timestamp, data in tqdm(reader.messages(connections=connections)):
            try:
                msg = deserialize_cdr(data, conn.msgtype)
            except:
                print('Error: Unable to deserialize messages from desired topic.')
                sys.exit(3)
            traj['timestamp'].append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
            traj['x'].append(msg.pose.pose.position.x)
            traj['y'].append(msg.pose.pose.position.y)
            traj['z'].append(msg.pose.pose.position.z)
            traj['q_x'].append(msg.pose.pose.orientation.x)
            traj['q_y'].append(msg.pose.pose.orientation.y)
            traj['q_z'].append(msg.pose.pose.orientation.z)
            traj['q_w'].append(msg.pose.pose.orientation.w)

        df = pd.DataFrame(traj)
        if not os.path.isdir(str(save_path)): os.makedirs(str(save_path))
        output_filepath = save_path / f'{output_name}.csv'
        df.to_csv(output_filepath, index=False, sep=' ', header=None)
        print(f'Done. Saved trajectory to {output_filepath}')


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str, required=True,
                        help='Name of input rosbag.')
    parser.add_argument('-t', '--topic',
                        type=str, required=True,
                        help='Topic to save (need to be Odometry msgs).')
    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Output name for csv file in TUM format. Will be saved in ‘trajectories’ folder.')
    return parser
if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    main(args.input, args.output, args.topic)