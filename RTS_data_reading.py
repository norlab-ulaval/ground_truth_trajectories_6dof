import os
import sys
import argparse
import numpy as np
import pandas as pd
import rosbags
from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from pathlib import Path
from tqdm import tqdm
from scipy import interpolate
import datetime

#----------------- Functions ----------------------
def convert_bag_to_csv(path, output):
    date = os.path.basename(path)
    input_file = os.path.join(path, f"{date}_inter_prism.bag")
    print('Converting rosbag into csv file:', path, input_file.split('.')[0] + '.csv')
    data = []
    with AnyReader([Path(input)]) as reader:
        connections = [x for x in reader.connections if x.topic == "/theodolite_master/theodolite_data"]
        for connection, _, rawdata in tqdm(reader.messages(connections=connections)):
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            data.append([timestamp, msg.theodolite_id, msg.distance, msg.azimuth, msg.elevation])
    df = pd.DataFrame(data, columns=['timestamp', 'id', 'distance', 'ha', 'va'])
    df.to_csv(output, index=False)
    if df.empty:
        print('No data found!')
    return df

def read_data(path):
    date = os.path.basename(path)
    csv_file = os.path.join(path, f"{date}_inter_prism.csv")
    bag_file = os.path.join(path, f"{date}_inter_prism.bag")
    if not os.path.exists(csv_file):
        df = convert_bag_to_csv(path, bag_file)
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        print('\nInvalid input file')
    if df.empty:
        print('No data found!')
    return df

def filter_data(df, distance, horizontal_angle, vertical_angle, split):
    df[['distance_diff', 'ha_diff', 'va_diff', 'timestamp_diff']] = df.groupby('id')[['distance', 'ha', 'va', 'timestamp']].diff()
    df = df[(df['distance_diff'] < distance) & (df['ha_diff'] < np.deg2rad(horizontal_angle)) & (df['va_diff'] < np.deg2rad(vertical_angle)) & (df['timestamp_diff'] < split)].copy()
    df.drop(columns=['distance_diff', 'ha_diff', 'va_diff', 'timestamp_diff'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('Data filtered!')
    return df

def data_to_3D_points_rad(df):
    df['X'] = (df['distance'] + 0.01) * np.cos(np.pi/2-df['ha']) * np.sin((df['va']))
    df['Y'] = (df['distance'] + 0.01) * np.sin(np.pi/2-df['ha']) * np.sin((df['va']))
    df['Z'] = (df['distance']+0.01) * np.cos((df['va']))
    return df

def data_to_3D_points_deg_to_rad(df):
    df['X'] = (df['distance'] + 0.01) * np.cos(np.deg2rad(90-df['ha'])) * np.sin(np.deg2rad(df['va']))
    df['Y'] = (df['distance'] + 0.01) * np.sin(np.deg2rad(90-df['ha'])) * np.sin(np.deg2rad(df['va']))
    df['Z'] = (df['distance']+0.01) * np.cos(np.deg2rad(df['va']))
    return df

def save_data(df, path, output, filtering):
    date = os.path.basename(path)
    print('Saving...')
    if filtering == True:
        output_file = os.path.join(output, f"{date}_RTS-data_filtered.csv")
        df.to_csv(output_file, index=False)
    else:
        output_file = os.path.join(output, f"{date}_RTS-data.csv")
        df.to_csv(output_file, index=False)
    print('Data saved to csv file:', output_file)

#----------------- Main ----------------------
def main(path, output, filtering, distance, horizontal_angle, vertical_angle, split, save, verbose, debug):
    if verbose:
        print('Path:', path)
        print('Output:', output)
        print('Filtering:', filtering)
        print('Distance:', distance)
        print('Horizontal angle:', horizontal_angle, 'deg/s')
        print('Vertical angle:', vertical_angle, 'deg/s')
        print('Split:', split, 's')
        print('Save:', save)
        print('Verbose:', verbose)
        print('Debug:', debug)
    df = read_data(path)
    assert filtering is not None, '--filtering: Apply filtreing or not, not provided'
    if filtering == True:
        assert distance is not None, '--distance: Distance threshold not provided'
        assert horizontal_angle is not None, '--horizontal_angle: Horizontal angle threshold not provided'
        assert vertical_angle is not None, '--vertical_angle: Vertical angle threshold not provided'
        assert split is not None, '--split: Split measurements not provided'

        print('\nFiltering...')
        print(' Distance:', distance, 'm/s')
        print(' Horizontal angle:', horizontal_angle, 'deg/s')
        print(' Vertical angle:', vertical_angle, 'deg/s')
        print(' Split:', split, 's')

        df = filter_data(df, distance, horizontal_angle, vertical_angle, split)

    df = data_to_3D_points_rad(df)
    df.drop(columns=['distance', 'ha', 'va'], inplace=True)

    if save == True:
        save_data(df, path, output, filtering)
    return df

def init_argparse():
    parser = argparse.ArgumentParser()

    #---------------- Input/Output ---------------------
    parser.add_argument('-p', '--path', 
                        type=dir_path, required=False,
                        help='Specify a path for the input data path')
    parser.add_argument('-o', '--output', 
                        type=str, required=False,
                        help='Specify a path for the output file, default is output/')
    parser.add_argument('-s', '--save',
                        action='store_true',
                        help='Save the output file')
    
    #----------------- Parameters ----------------------
    parser.add_argument('-f', '--filtering', 
                        action='store_true',
                        help='Apply filtering or not') #if filtering is True, then parameters are required
    parser.add_argument("--distance",
                        type=float, required=False,
                        help="Distance threshold [m/s]")
    parser.add_argument("--horizontal_angle",
                        type=float, required=False, 
                        help="Horizontal angle threshold [deg/s]")
    parser.add_argument("--vertical_angle",
                        type=float, required=False,
                        help="Vertical angle threshold [deg/s]")
    parser.add_argument('--split',
                        type=float, required=False,
                        help='Split measurements [s]')
    
    #--------------- Verbose/Debug ------------------
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Full pipeline for computing rosbags into ground truth trajectory.')
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Debug mode. Print all debug messages.')
    return parser

#--------------- Types ------------------
def dir_path(path):
    if os.path.isdir(path):
        if path.endswith('/'):
            path = path[:-1]
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.path, 'output')
    main(args.path, args.output, args.filtering, args.distance, args.horizontal_angle, args.vertical_angle, args.split, args.save, args.verbose, args.debug)