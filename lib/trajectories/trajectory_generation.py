import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
BASE_PATH = Path(__file__).absolute().parents[1] 
import argparse
import point_to_point as ptp
from scipy.spatial.transform import Rotation as R


def split_rts_gps(df_gps, df_rts):

    df_gps['Timestamp'] = df_gps['Timestamp'].values.astype(np.float64)
    
    # Merge the two dataframes by timestamp
    merged_data = pd.merge_asof(
        df_rts[['Timestamp', 'T']],
        df_gps[['Timestamp', 'T']],
        on=['Timestamp'], tolerance=0.5, direction='nearest'
    ).dropna()

    # Find points that are only in the RTS trajectory
    merged_rts_common = pd.merge(
        merged_data[['Timestamp', 'T_x']],
        df_rts[['Timestamp', 'T']],
        on=['Timestamp'],
        how='outer',
        indicator=True
    )
    rts_only_traj = merged_rts_common[merged_rts_common['_merge'] == 'right_only']

    # Merge the two dataframes by timestamp in the other direction
    merged_data = pd.merge_asof(
        df_gps[['Timestamp', 'T']],
        df_rts[['Timestamp', 'T']],
        on=['Timestamp'], tolerance=0.2, direction='nearest'
    ).dropna()

    # Find points that are only in the GPS trajectory
    merged_gps_common = pd.merge(
        merged_data[['Timestamp', 'T_x']],
        df_gps[['Timestamp', 'T']],
        on=['Timestamp'], how='outer', indicator=True
    )
    gps_only_traj = merged_gps_common[merged_gps_common['_merge'] == 'right_only']

    
    # Clean the dataframes
    common_traj = merged_data.rename(columns={'T_x':'T_gps', 'T_y':'T_rts'})
    rts_only_traj.drop(['T_x', '_merge'], axis=1, inplace=True)
    gps_only_traj.drop(['T_x', '_merge'], axis=1, inplace=True)
    
    return gps_only_traj, rts_only_traj, common_traj


def generate_reconstructed_traj(gps_only_traj, rts_only_traj, common_traj):

    # Transform the common trajectory to the GPS frame
    P_common = np.array([
        [T[0,3] for T in common_traj['T_rts']], 
        [T[1,3] for T in common_traj['T_rts']],
        [T[2,3] for T in common_traj['T_rts']],
        np.ones(len(common_traj['T_rts']))
    ])
    Q_common = np.array([
        [T[0,3] for T in common_traj['T_gps']],
        [T[1,3] for T in common_traj['T_gps']],
        [T[2,3] for T in common_traj['T_gps']],
        np.ones(len(common_traj['T_gps']))
    ])
    T = ptp.minimization(P_common, Q_common)
    common_traj['T_rts'] = common_traj['T_rts'].apply(lambda x: T @ x)
    rts_only_traj['T'] = rts_only_traj['T'].apply(lambda x: T @ x)

    # Concat the common, gps only and rts only trajectories
    common_traj.rename(columns={'T_rts':'T'}, inplace=True)
    common_traj.drop(['T_gps'], axis=1, inplace=True)
    reconstructed_traj = pd.concat([common_traj, gps_only_traj, rts_only_traj], ignore_index=True)
    reconstructed_traj.sort_values(by=['Timestamp'], inplace=True)

    return reconstructed_traj


def pose_quat_to_tranform(dataframe):

    transforms = []
    for idx, pose in dataframe.iterrows():
        r = R.from_quat([pose['qx'], pose['qy'], pose['qz'], pose['qw']])
        R_mat = r.as_matrix()
        T = np.eye(4)
        T[:3,:3] = R_mat
        T[:3,3] = np.array([pose['X'], pose['Y'], pose['Z']])
        transforms.append(T)

    dataframe['T'] = transforms
    dataframe.drop(['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw'], axis=1, inplace=True)
    return dataframe


def transform_to_pose_quat(dataframe):

    dataframe['X'] = [T[0,3] for T in dataframe['T']]
    dataframe['Y'] = [T[1,3] for T in dataframe['T']]
    dataframe['Z'] = [T[2,3] for T in dataframe['T']]
    rot = [R.from_matrix(T[:3,:3]) for T in dataframe['T']]
    dataframe['qx'] = [r.as_quat()[0] for r in rot]
    dataframe['qy'] = [r.as_quat()[1] for r in rot]
    dataframe['qz'] = [r.as_quat()[2] for r in rot]
    dataframe['qw'] = [r.as_quat()[3] for r in rot]
    dataframe.drop(['T'], axis=1, inplace=True)
    return dataframe


def plot_reconstructed_traj(gps_only_traj, rts_only_traj, common_traj, reconstructed_traj, arrows=False):

    fig ,axs = plt.subplots(2,2,figsize =(8,8))
    colors = ['lightgrey', 'lightskyblue', 'lightsalmon', 'red']
    labels = ['RTS only', 'GPS only', 'Common', 'Reconstructed']
    trajs = [rts_only_traj, gps_only_traj, common_traj, reconstructed_traj]
    for ax, traj, color, label in zip(axs.flat, trajs, colors, labels):
        X = [T[0,3] for T in traj['T']]
        Y = [T[1,3] for T in traj['T']]
        ax.scatter(X, Y, s=2, c=color, label = label)
        ax.title.set_text(label + ' trajectory')
        ax.set_aspect('equal')
        ax.legend()

        if arrows:
            nx_u = [T[0,0] for T in traj['T']]
            nx_v = [T[1,0] for T in traj['T']]
            ny_u = [T[0,1] for T in traj['T']]
            ny_v = [T[1,1] for T in traj['T']]
            ax.quiver(X, Y, nx_u, nx_v, color='r', scale=1, scale_units='xy', angles='xy', headwidth=1)
            ax.quiver(X, Y, ny_u, ny_v, color='g', scale=1, scale_units='xy', angles='xy', headwidth=1)

    plt.show()

def plot_icp(df_icp):
    fig, ax = plt.subplots(1,1,figsize =(8,8))
    X = [T[0,3] for T in df_icp['T']]
    Y = [T[1,3] for T in df_icp['T']]
    ax.scatter(X, Y, s=2, c='lightgrey', label = 'ICP')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()


def find_transform_between_origins(gt_traj, df_icp):

    merged_data = pd.merge_asof(
        gt_traj[['Timestamp', 'T']],
        df_icp[['Timestamp', 'T']],
        on=['Timestamp'], tolerance=0.5, direction='nearest'
    ).dropna()

    #T icp vers gps
    unit_vectors = np.eye(3)
    unit_vectors = np.vstack((unit_vectors, np.array([1,1,1])))

    P_gt = merged_data['T_x'].iloc[0] @ unit_vectors
    P_icp = merged_data['T_y'].iloc[0] @ unit_vectors
    T = ptp.minimization(P_icp, P_gt)

    return T


def plot_icp_vs_gt(icp_traj, gt_traj, arrows=False):

    fig, ax = plt.subplots(1,1,figsize =(8,8))
    colors = ['lightgrey', 'lightsalmon']
    labels = ['ICP', 'Ground truth']
    trajs = [icp_traj, gt_traj]
    for traj, color, label in zip(trajs, colors, labels):
        X = [T[0,3] for T in traj['T']]
        Y = [T[1,3] for T in traj['T']]
        if arrows:
            nx_u = [T[0,0] for T in traj['T']]
            nx_v = [T[1,0] for T in traj['T']]
            ny_u = [T[0,1] for T in traj['T']]
            ny_v = [T[1,1] for T in traj['T']]
            ax.quiver(X, Y, nx_u, nx_v, color='r', scale=1, scale_units='xy', angles='xy', headwidth=1)
            ax.quiver(X, Y, ny_u, ny_v, color='g', scale=1, scale_units='xy', angles='xy', headwidth=1)

        ax.scatter(X, Y, s=2, c=color, label = label)
        ax.set_aspect('equal')
        ax.legend()

    plt.show()


def main(gps_file, rts_file, icp_file, plot, arrows, save):

    df_gps = pd.read_csv(BASE_PATH / f'{gps_file}', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
    df_rts = pd.read_csv(BASE_PATH / f'{rts_file}', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
    df_icp = pd.read_csv(BASE_PATH / f'{icp_file}', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')

    for df in [df_gps, df_rts, df_icp]:
        df = pose_quat_to_tranform(df)
    gps_only_traj, rts_only_traj, common_traj = split_rts_gps(df_gps, df_rts)
    reconstructed_traj = generate_reconstructed_traj(gps_only_traj, rts_only_traj, common_traj)
    plot_icp(df_icp)
    if plot: 
        plot_reconstructed_traj(gps_only_traj, rts_only_traj, common_traj, reconstructed_traj, arrows=arrows)
    # reconstructed_traj.to_csv(BASE_PATH / 'trajectories' / 'gt_reconstructed_traj1.csv', index=False, header=False, sep = ' ')

    T_icp_gt = find_transform_between_origins(reconstructed_traj, df_icp)
    df_icp['T'] = df_icp['T'].apply(lambda x: T_icp_gt @ x)
    if plot:
        plot_icp_vs_gt(df_icp, reconstructed_traj, arrows=arrows)

    if save:
        df_icp = transform_to_pose_quat(df_icp)
        reconstructed_traj = transform_to_pose_quat(reconstructed_traj)
        df_icp.to_csv(BASE_PATH / 'output' / 'icp_reconstructed_traj.csv', index=False, header=False, sep = ' ')
        reconstructed_traj.to_csv(BASE_PATH / 'output' / 'gt_reconstructed_traj.csv', index=False, header=False, sep = ' ')


def init_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('-gps', '--gps',
                        type=str, required=True,
                        help='Name of gps file.')
    parser.add_argument('-rts', '--rts',
                        type=str, required=True,
                        help='Name of rts file.')
    parser.add_argument('-icp', '--icp',
                        type=str, required=True,
                        help='Name of icp file.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot trajectories for debugging.')
    parser.add_argument('-a', '--arrows', action='store_true',
                        help='Plot arrows onto trajectories.')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save resulting trajectories in output folder.')
    return parser


if __name__ == '__main__':

    parser = init_argparse()
    args = parser.parse_args()
    main(args.gps, args.rts, args.icp, args.plot, args.arrows, args.save)