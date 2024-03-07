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
def convert_bag_to_csv(input, output):
    print('Converting rosbag into csv file:', input, input.split('.')[0] + '.csv')
    data = []
    with AnyReader([Path(input)]) as reader:
        connections = [x for x in reader.connections if x.topic == "/theodolite_master/theodolite_data"]
        for connection, _, rawdata in tqdm(reader.messages(connections=connections)):
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            data.append([timestamp, msg.theodolite_id, msg.distance, msg.azimuth, msg.elevation])
    df = pd.DataFrame(data, columns=['timestamp', 'id', 'distance', 'ha', 'va'])
    df.to_csv(output, index=False)
    return df

def read_data(input):
    _, extension = os.path.splitext(input)
    if extension == '.bag':
        print('\nReading bag...')
        df = convert_bag_to_csv(input, input.split('.')[0] + '.csv')
    elif extension == '.csv':
        print('\nReading csv...')
        df = pd.read_csv(input)
    else:
        print('\nInvalid input file')
    return df

def filter_data(df, distance, horizontal_angle, vertical_angle, split):
    df[['distance_diff', 'ha_diff', 'va_diff', 'timestamp_diff']] = df.groupby('id')[['distance', 'ha', 'va', 'timestamp']].diff()
    df = df[(df['distance_diff'] < distance) & (df['ha_diff'] < np.deg2rad(horizontal_angle)) & (df['va_diff'] < np.deg2rad(vertical_angle)) & (df['timestamp_diff'] < split)].copy()
    df.drop(columns=['distance_diff', 'ha_diff', 'va_diff', 'timestamp_diff'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print('Data filtered!')
    return df

def data_to_3D_points(df):
    df['X'] = (df['distance'] + 0.01) * np.cos(np.deg2rad(90-df['ha'])) * np.sin(np.deg2rad(df['va']))
    df['Y'] = (df['distance'] + 0.01) * np.sin(np.deg2rad(90-df['ha'])) * np.sin(np.deg2rad(df['va']))
    df['Z'] = (df['distance']+0.01) * np.cos(np.deg2rad(df['va']))
    print('Raw data converted to 3D points!')
    return df

def save_data(df, input, output, filtering):
    date = os.path.basename(input)[:8]
    print('Saving...')
    if filtering == True:
        output_file = os.path.join(output, f"{date}-ground_truth_filtered.csv")
        df.to_csv(output_file, index=True)
    else:
        output_file = os.path.join(output, f"{date}-ground_truth.csv")
        df.to_csv(output_file, index=True)
    print('Data saved to csv file:', output_file)

#----------------- Main ----------------------
def main(input, output, filtering, distance, horizontal_angle, vertical_angle, split, save, verbose, debug):
    if verbose:
        print('Input:', input)
        print('Output:', output)
        print('Filtering:', filtering)
        print('Distance:', distance)
        print('Horizontal angle:', horizontal_angle, 'deg/s')
        print('Vertical angle:', vertical_angle, 'deg/s')
        print('Split:', split, 's')
        print('Save:', save)
        print('Verbose:', verbose)
        print('Debug:', debug)
    df = read_data(input)

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

    df = data_to_3D_points(df)

    if save == True:
        save_data(df, input, output, filtering)
    return df

def init_argparse():
    parser = argparse.ArgumentParser()

    #---------------- Input/Output ---------------------
    parser.add_argument('-i', '--input', 
                        type=str, required=False,
                        help='Specify a path for the input path')
    parser.add_argument('-o', '--output', 
                        type=str, required=False, default=os.path.join(os.path.expanduser('~'), 'repos', 'ground_truth_generation', 'output/'),
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

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    main(args.input, args.output, args.filtering, args.distance, args.horizontal_angle, args.vertical_angle, args.split, args.save, args.verbose, args.debug)


