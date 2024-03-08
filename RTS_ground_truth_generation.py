import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# def read_sensor_positions(file_path, file_name_output, name_lidar):
#     points = []
#     with open(file_path, "r") as file:
#         for line in file:
#             item = line.split(" ")
#             ha = float(item[0]) + float(item[1]) * 1 / 60 + float(item[2]) * 1 / 3600
#             va = float(item[6]) + float(item[7]) * 1 / 60 + float(item[8]) * 1 / 3600
#             d = float(item[12])
#             points.append(coordinates_transformation(d, ha, va, 'deg'))

#     P1, P2, P3, G1, G2, G3 = points[:6]

#     if len(points) > 6 and name_lidar == "Robosense_32":
#         distance_lidar_top_to_lidar_origin = 0.063  # In meter, for RS32 on warthog
#         L1, L2, L3 = points[6:9]
#         u = L2[:3] - L1[:3]
#         v = L3[:3] - L1[:3]
#         A = np.array([[u[0], u[1]], [v[0], v[1]]])
#         y = np.array([-u[2], -v[2]])
#         x = np.linalg.solve(A, y)
#         n = np.array([x[0], x[1], 1])
#         I = L2[:3] + np.dot(L1[:3] - L2[:3], L3[:3] - L2[:3]) / np.dot(L3[:3] - L2[:3], L3[:3] - L2[:3]) * (L3[:3] - L2[:3])
#         z = I - L1[:3]
#         z_unit = 1 / np.linalg.norm(z) * z
#         O = distance_lidar_top_to_lidar_origin * z_unit + L1[:3]
#         Ox = -1 / np.linalg.norm(n) * n + O
#         Oz = -1 * z_unit + O
#         Oy = 1 * np.cross(-z_unit, -1 / np.linalg.norm(n) * n) + O

#     with open(file_name_output, "w+") as csv_file:
#         csv_file.write(f"{P1[0]} {P1[1]} {P1[2]} 1\n")
#         csv_file.write(f"{P2[0]} {P2[1]} {P2[2]} 1\n")
#         csv_file.write(f"{P3[0]} {P3[1]} {P3[2]} 1\n")
#         csv_file.write(f"{G1[0]} {G1[1]} {G1[2]} 1\n")
#         csv_file.write(f"{G2[0]} {G2[1]} {G2[2]} 1\n")
#         csv_file.write(f"{G3[0]} {G3[1]} {G3[2]} 1\n")
#         if len(points) > 6 and name_lidar == "Robosense_32":
#             csv_file.write(f"{O[0]} {O[1]} {O[2]} 1\n")
#             csv_file.write(f"{Ox[0]} {Ox[1]} {Ox[2]} 1\n")
#             csv_file.write(f"{Oy[0]} {Oy[1]} {Oy[2]} 1\n")
#             csv_file.write(f"{Oz[0]} {Oz[1]} {Oz[2]} 1\n")
#     print('\nSensors coordinates in theodolite frame :')
#     print(f"Prism 1: {P1}")
#     print(f"Prism 2: {P2}")
#     print(f"Prism 3: {P3}")
#     print(f"GNSS 1: {G1}")
#     print(f"GNSS 2: {G2}")
#     print(f"GNSS 3: {G3}")
#     if len(points) > 6 and name_lidar == "Robosense_32":
#         print('\nLiDAR center :')
#         print(f"O: {O}")
#         print(f"Ox: {Ox}")
#         print(f"Oy: {Oy}")
#         print(f"Oz: {Oz}")






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

def read_ground_control_points(path):
    input = os.path.join(path, 'GCP.txt')
    df_ground_control_points = pd.read_csv(input, delimiter=' ,', names=['id', 'marker', 'status', 'ha', 'va', 'distance', 'sec', 'nsec'], skiprows=[0], engine='python')
    if df_ground_control_points.empty:
        print('No ground control points found!')
    return df_ground_control_points

def calibration(path, output, save):
    date = os.path.basename(path)
    input = os.path.join(path, 'calibration_raw.csv')
    if not os.path.exists(input):
        print('No calibration file found!')
    df = pd.read_csv(input, delimiter=' ', usecols=[0, 1, 2, 6, 7, 8, 12], names=['ha_deg', 'ha_min', 'ha_sec', 'va_deg', 'va_min', 'va_sec', 'distance'])
    if df.empty:
        print('No calibration data found!')
    df_calibration = pd.DataFrame(columns=['ha', 'va', 'd'])
    for rows in df:
        df_calibration['ha'] = df['ha_deg'] + df['ha_min'] * 1 / 60 + df['ha_sec'] * 1 / 3600
        df_calibration['va'] = df['va_deg'] + df['va_min'] * 1 / 60 + df['va_sec'] * 1 / 3600
        df_calibration['d'] = df['distance']
    # df.rename(index=)
    print(df_calibration)
    if save == True:
        df_calibration.to_csv(output, f"{date}_calibration_file.csv", index=True)
    return df_calibration
    
    
def save_data(df, path, output):
    date = os.path.basename(path)
    print('Saving...')
    output_file = os.path.join(output, f"RTS-ground_truth-{date}.csv")
    df.to_csv(output_file, index=True)
    print('Data saved to csv file:', output_file)

#----------------- Main ----------------------
def main(path, output, save, verbose, debug):
    if verbose:
        print('Path:', path)
        print('Output:', output)
        print('Save:', save)
        print('Verbose:', verbose)
        print('Debug:', debug)
    df = read_data(path, debug)
    df_ground_control_points = read_ground_control_points(path)
    # df_calibration = calibration(path, output, save)

    # if save == True:
    #     save_data(df, input_path, output_path)
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
    main(args.path, args.output, args.save, args.verbose, args.debug)