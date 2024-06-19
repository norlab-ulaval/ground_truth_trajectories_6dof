import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from RTS_data_reading import data_to_3D_points_rad, data_to_3D_points_deg_to_rad
from point_to_point import *
from visualize import *

#----------------- Constants ----------------------
DISTANCE_LIDAR_TOP_TO_LIDAR_ORIGIN = 0.063 # In meter, for RS32 on warthog

#----------------- Calibration Functions ----------------------
def compute_transformation_to_lidar_frame(df_calibration, distance_lidar_top_to_lidar_origin, debug):
    L1 = df_calibration.loc['lidar_top']
    L2 = df_calibration.loc['lidar_left']
    L3 = df_calibration.loc['lidar_right']
    u = L2 - L1
    v = L3 - L1
    A = np.array([[u.iloc[0], u.iloc[1]], [v.iloc[0], v.iloc[1]]])
    y = np.array([-u.iloc[2], -v.iloc[2]])
    x = np.linalg.solve(A, y)
    n = np.array([x[0], x[1], 1])
    I = L2 + np.dot(L1 - L2, L3 - L2) / np.dot(L3 - L2, L3 - L2) * (L3 - L2)
    z = I - L1
    z_unit = 1 / np.linalg.norm(z) * z
    df_calibration.loc['lidar_origin'] = distance_lidar_top_to_lidar_origin * z_unit + L1
    df_calibration.loc['lidar_x'] = -1 / np.linalg.norm(n) * n + df_calibration.loc['lidar_origin']
    df_calibration.loc['lidar_z'] = -1 * z_unit + df_calibration.loc['lidar_origin']
    df_calibration.loc['lidar_y'] = 1 * np.cross(-z_unit, -1 / np.linalg.norm(n) * n) + df_calibration.loc['lidar_origin']
    O = df_calibration.loc['lidar_origin']
    nx = df_calibration.loc['lidar_x'] - df_calibration.loc['lidar_origin']
    ny = df_calibration.loc['lidar_y'] - df_calibration.loc['lidar_origin']
    nz = df_calibration.loc['lidar_z'] - df_calibration.loc['lidar_origin']
    T_from_theodolite_to_lidar = np.eye(4)
    T_from_theodolite_to_lidar = np.array([[nx.iloc[0],ny.iloc[0],nz.iloc[0],O.iloc[0]],
                                                [nx.iloc[1],ny.iloc[1],nz.iloc[1],O.iloc[1]],
                                                [nx.iloc[2],ny.iloc[2],nz.iloc[2],O.iloc[2]],
                                                [0,0,0,1]])
    T_from_theodolite_to_lidar[:3, 3] = df_calibration.loc['lidar_origin']
    T_from_theodolite_to_lidar = np.linalg.inv(T_from_theodolite_to_lidar)

    df_calibration_numpy = df_calibration.to_numpy()
    df_calibration_numpy = np.append(df_calibration_numpy, np.ones((df_calibration_numpy.shape[0], 1)), axis=1)
    df_calibration_numpy = T_from_theodolite_to_lidar @ df_calibration_numpy.T

    df_calibration = pd.DataFrame(df_calibration_numpy[:3, :].T, columns=df_calibration.columns, index=df_calibration.index)
    df_calibration.drop(index=['lidar_left', 'lidar_right', 'lidar_top', 'lidar_x', 'lidar_y', 'lidar_z'], inplace=True)
    if debug == True:
        print('Transformation matrix of theodolite to LiDAR frame:\n', T_from_theodolite_to_lidar)
        print('\nSensors coordinates in LiDAR frame:', df_calibration)
    return df_calibration,T_from_theodolite_to_lidar

#----------------- Functions ----------------------
def read_data(path, debug):
    date = os.path.basename(path)
    input = os.path.join(path, 'output', f"{date}_RTS-data.csv")
    if not os.path.exists(input):
        input = os.path.join(path, 'output', f"{date}_RTS-data_filtered.csv")
        df = pd.read_csv(input)
    elif os.path.exists(input):
        df = pd.read_csv(input)
    else :
        print('\nInvalid input file')
    if df.empty:
        print('No data found!')
    if debug == True:
        print(date)
        print(input)
        print(df)
    return df

def data_to_3D_points_deg_to_rad_calibration(df):
    df['X'] = df['distance'] * np.cos(-df['ha'] * np.pi / 180) * np.cos((90 - df['va']) * np.pi / 180)
    df['Y'] = df['distance'] * np.sin(-df['ha'] * np.pi / 180) * np.cos((90 - df['va']) * np.pi / 180)
    df['Z'] = df['distance'] * np.sin((90 - df['va']) * np.pi / 180)
    return df

def calibration(path, output, save, distance_lidar_top_to_lidar_origin, debug):
    date = os.path.basename(path)
    input = os.path.join(path, 'calibration_raw.csv')
    if not os.path.exists(input):
        print('No calibration file found!')
    df = pd.read_csv(input, delimiter=' ', usecols=[0, 1, 2, 6, 7, 8, 12], names=['ha_deg', 'ha_min', 'ha_sec', 'va_deg', 'va_min', 'va_sec', 'distance'])
    if df.empty:
        print('No calibration data found!')
    df_calibration_raw = pd.DataFrame(columns=['ha', 'va', 'distance'])
    for rows in df:
        df_calibration_raw['ha'] = df['ha_deg'] + df['ha_min'] * 1 / 60 + df['ha_sec'] * 1 / 3600
        df_calibration_raw['va'] = df['va_deg'] + df['va_min'] * 1 / 60 + df['va_sec'] * 1 / 3600
        df_calibration_raw['distance'] = df['distance']
    df_calibration_raw = df_calibration_raw
    df_calibration = data_to_3D_points_deg_to_rad_calibration(df_calibration_raw)
    df_calibration.drop(columns=['ha', 'va', 'distance'], inplace=True)
    df_calibration['id'] = ['prism1', 'prism2', 'prism3', 'gnss1', 'gnss2', 'gnss3', 'lidar_top', 'lidar_left', 'lidar_right']
    df_calibration.set_index('id', inplace=True)
    if debug == True:
        print("Sensors data for calibration:", df_calibration)   
    df_calibration = compute_transformation_to_lidar_frame(df_calibration, distance_lidar_top_to_lidar_origin, debug)
    return df_calibration

def read_ground_control_points(path, debug):
    input = os.path.join(path, 'GCP.txt')
    df_ground_control_points = pd.read_csv(input, delimiter=' ,', names=['id', 'marker', 'status', 'ha', 'va', 'distance', 'sec', 'nsec'], skiprows=[0], engine='python')
    if df_ground_control_points.empty:
        print('No ground control points found!')
    df_ground_control_points = data_to_3D_points_rad(df_ground_control_points)
    df_ground_control_points.drop(columns=['marker', 'ha', 'va', 'distance' ,'sec', 'nsec'], inplace=True)
    P1, P2, P3, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1 = compute_theodolite_Transform(df_ground_control_points, reference_frame=1)
    df_list = []
    for frame in [P1, P2, P3]:
        df_temp = pd.DataFrame(columns=['X', 'Y', 'Z'])
        df_temp['X'] = frame.T[:, 0]
        df_temp['Y'] = frame.T[:, 1]
        df_temp['Z'] = frame.T[:, 2]
        df_list.append(df_temp)
    df_ground_control_points_frame_1 = pd.concat(df_list)
    if debug == True:
        print("Ground control point 3D point:", df_ground_control_points_frame_1)
    return df_ground_control_points_frame_1, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1

def generate_ground_truth(df, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1, df_calibration, debug):
    df_ground_truth_theodolite_1 = apply_theodolite_Transform_to_data(df, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1)
    df_ground_truth_lidar_frame = generate_ground_truth_lidar_frame(df_ground_truth_theodolite_1, df_calibration)
    if debug == True:
        print('Ground truth df in lidar frame:', df_ground_truth_lidar_frame)
    return df_ground_truth_lidar_frame
    
def save_data(df, df_calibration, df_ground_control_points_frame_1, path, output):
    date = os.path.basename(path)
    print('Saving...')
    output_file = os.path.join(output, f"RTS-ground_truth-{date}.csv")
    df.to_csv(output_file, index=False)
    df_calibration.to_csv(os.path.join(output, f"calibration_file.csv"), index=True)
    df_ground_control_points_frame_1.to_csv(os.path.join(output, f"ground_control_points.csv"), index=False)
    print('Data saved to csv file:', output_file)
    print('Calibration data saved to csv file:', os.path.join(output, f"calibration_file.csv"))
    print('Ground control points saved to csv file:', os.path.join(output, f"ground_control_points.csv"))

#----------------- Main ----------------------
def main(path, output, save, verbose, debug, display):
    if verbose:
        print('Path:', path)
        print('Output:', output)
        print('Save:', save)
        print('Verbose:', verbose)
        print('Debug:', debug)
        print('Display:', display)

    df = read_data(path, debug)

    df_calibration, T_from_theodolite_to_lidar = calibration(path, output, save, DISTANCE_LIDAR_TOP_TO_LIDAR_ORIGIN,debug)

    df_ground_control_points_frame_1, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1 = read_ground_control_points(path, debug)

    # df_ground_truth_theodolite_1 = generate_ground_truth(df, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1, df_calibration, debug)

    if display == True:
        display_calibration_data(df_calibration)
        display_ground_control_points(df_ground_control_points_frame_1, 'Ground control points')
        display_ground_truth(df)
        plt.show()

    if save == True:
        save_data(df, df_calibration, df_ground_control_points_frame_1, path, output)
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
    
    #--------------- Verbose/Debug ------------------
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Full pipeline for computing rosbags into ground truth trajectory.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug mode. Print all debug messages.')
    parser.add_argument('-d', '--display',
                        action='store_true',
                        help='Display the 3D plot of the calibration data, ...')
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
    main(args.path, args.output, args.save, args.verbose, args.debug, args.display)