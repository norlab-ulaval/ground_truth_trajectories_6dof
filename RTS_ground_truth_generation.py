import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from RTS_data_reading import data_to_3D_points
import matplotlib.pyplot as plt
from point_to_point import minimization

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

    nx = df_calibration.loc['lidar_x'] - df_calibration.loc['lidar_origin']
    ny = df_calibration.loc['lidar_y'] - df_calibration.loc['lidar_origin']
    nz = df_calibration.loc['lidar_z'] - df_calibration.loc['lidar_origin']

    T = np.eye(4)
    T[:3, :3] = np.array([nx, ny, nz])  # Reshape the vectors to match the dimensions of the transformation matrix
    T[:3, 3] = df_calibration.loc['lidar_origin']
    T = np.linalg.inv(T)

    df_calibration_numpy = df_calibration.to_numpy()
    df_calibration_numpy = np.append(df_calibration_numpy, np.ones((df_calibration_numpy.shape[0], 1)), axis=1)
    df_calibration_numpy = T @ df_calibration_numpy.T

    df_calibration = pd.DataFrame(df_calibration_numpy[:3, :].T, columns=df_calibration.columns, index=df_calibration.index)
    # df_calibration = pd.DataFrame(df_calibration_numpy[:,:3], columns=df_calibration.columns, index=df_calibration.index)
    df_calibration.drop(index=['lidar_left', 'lidar_right', 'lidar_top'], inplace=True)
    if debug == True:
        print('Transformation matrix of theodolite to LiDAR frame :\n', T)
        print('\nSensors coordinates in LiDAR frame :', df_calibration)
    return df_calibration

#----------------- Display Functions ----------------------

def display_calibration_data(df_calibration):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(df_calibration.loc['prism1']['X'], df_calibration.loc['prism1']['Y'], df_calibration.loc['prism1']['Z'], color = "C1")
    ax.scatter3D(df_calibration.loc['prism2']['X'], df_calibration.loc['prism2']['Y'], df_calibration.loc['prism2']['Z'], color = "C2")
    ax.scatter3D(df_calibration.loc['prism3']['X'], df_calibration.loc['prism3']['Y'], df_calibration.loc['prism3']['Z'], color = "C3")
    ax.scatter3D(df_calibration.loc['lidar_origin']['X'], df_calibration.loc['lidar_origin']['Y'], df_calibration.loc['lidar_origin']['Z'], color = "C0")

    ax.scatter3D(df_calibration.loc['gnss1']['X'], df_calibration.loc['gnss1']['Y'], df_calibration.loc['gnss1']['Z'], color = "C4")
    ax.scatter3D(df_calibration.loc['gnss2']['X'], df_calibration.loc['gnss2']['Y'], df_calibration.loc['gnss2']['Z'], color = "C5")
    ax.scatter3D(df_calibration.loc['gnss3']['X'], df_calibration.loc['gnss3']['Y'], df_calibration.loc['gnss3']['Z'], color = "C6")

    ax.text(df_calibration.loc['prism1']['X']-0.02, df_calibration.loc['prism1']['Y']+0.02, df_calibration.loc['prism1']['Z']+0.02, 'prism1')
    ax.text(df_calibration.loc['prism2']['X']-0.02, df_calibration.loc['prism2']['Y']+0.02, df_calibration.loc['prism2']['Z']+0.02, 'prism2')
    ax.text(df_calibration.loc['prism3']['X']-0.02, df_calibration.loc['prism3']['Y']+0.02, df_calibration.loc['prism3']['Z']+0.02, 'prism3')
    ax.text(df_calibration.loc['lidar_origin']['X']-0.02, df_calibration.loc['lidar_origin']['Y']+0.02, df_calibration.loc['lidar_origin']['Z']+0.02, 'lidar_origin')

    ax.text(df_calibration.loc['gnss1']['X']-0.02, df_calibration.loc['gnss1']['Y']+0.02, df_calibration.loc['gnss1']['Z']+0.02, 'gnss1')
    ax.text(df_calibration.loc['gnss2']['X']-0.02, df_calibration.loc['gnss2']['Y']+0.02, df_calibration.loc['gnss2']['Z']+0.02, 'gnss2')
    ax.text(df_calibration.loc['gnss3']['X']-0.02, df_calibration.loc['gnss3']['Y']+0.02, df_calibration.loc['gnss3']['Z']+0.02, 'gnss3')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Calibration data')

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

def read_ground_control_points(path, df):
    input = os.path.join(path, 'GCP.txt')
    df_ground_control_points = pd.read_csv(input, delimiter=' ,', names=['id', 'marker', 'status', 'ha', 'va', 'distance', 'sec', 'nsec'], skiprows=[0], engine='python')
    if df_ground_control_points.empty:
        print('No ground control points found!')
    return df

def generate_ground_truth(df, df_ground_control_points, debug):
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
    df_calibration = data_to_3D_points(df_calibration_raw)
    df_calibration.drop(columns=['ha', 'va', 'distance'], inplace=True)
    df_calibration['id'] = ['prism1', 'prism2', 'prism3', 'gnss1', 'gnss2', 'gnss3', 'lidar_top', 'lidar_left', 'lidar_right']
    df_calibration.set_index('id', inplace=True)
    if debug == True:
        print("Sensors data for calibration:", df_calibration)   
    df_calibration = compute_transformation_to_lidar_frame(df_calibration, distance_lidar_top_to_lidar_origin, debug)
    if save == True:
        df_calibration_raw.to_csv(output, f"{date}_calibration_file.csv", index=True)
    return df_calibration

def generate_ground_truth_pose(df, df_calibration,debug):
    #for df['id'] = 1 compute the transformation with df_calibration['id'] = 1

    prism1 = df_calibration.loc['prism1']
    prism2 = df_calibration.loc['prism2']
    prism3 = df_calibration.loc['prism3']
    df = df.to_numpy()
    # print(df[0][5:8])
    # print(np.transpose(df[0][5:8]))
    # print(prism1)
    # for index in df[:, 1]:
    #     if index == 1:
    #         print(df[5:8])
    #         df['T'] = minimization(df[5:8].T, prism1)
    # print(df)


        # df['T'] = minimization(df[['X','Y', 'Z']], df_RTS_calibration[['X','Y', 'Z']])
        # for index, row in df_id.iterrows():
        #     df.loc[index, 'T'] = minimization(row['X', 'Y', 'Z'], df_RTS_calibration_id)
    # print(df)
    # df['T'] = minimization(df.loc['X', 'Y', 'Z'], df_calibration)
    # df['T'] = minimization(df, df_calibration)
    # return df
    
def save_data(df, path, output):
    date = os.path.basename(path)
    print('Saving...')
    output_file = os.path.join(output, f"RTS-ground_truth-{date}.csv")
    df.to_csv(output_file, index=False)
    print('Data saved to csv file:', output_file)

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
    df_ground_control_points = read_ground_control_points(path, df)
    # df = generate_ground_truth(df, df_ground_control_points, debug)

    df_calibration = calibration(path, output, save, DISTANCE_LIDAR_TOP_TO_LIDAR_ORIGIN,debug)
    df = generate_ground_truth_pose(df, df_calibration, debug)

    if display == True:
        display_calibration_data(df_calibration)
        plt.show()

    if save == True:
        save_data(df, path, output)

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