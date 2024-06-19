import numpy as np
import pandas as pd

def minimization(P, Q):
    errors_before = Q - P # Errors at the beginning
    mu_p = np.mean(P[0:3, :], axis=1) #Centroide of each pointcloud
    mu_q = np.mean(Q[0:3, :], axis=1)
    P_mu = np.ones((3, P.shape[1])) # Centered each pointclouds
    Q_mu = np.ones((3, Q.shape[1]))
    for i in range(0, P_mu.shape[1]):
        P_mu[0:3, i] = P[0:3, i] - mu_p
    for i in range(0, Q_mu.shape[1]):
        Q_mu[0:3, i] = Q[0:3, i] - mu_q
    H = P_mu @ Q_mu.T # Cross covariance matrix
    U, s, V = np.linalg.svd(H) # Use SVD decomposition
    M = np.eye(3) # Compute rotation
    M[2, 2] = np.linalg.det(V.T @ U.T)
    R = V.T @ M @ U.T
    t = mu_q - R @ mu_p # Compute translation
    T = np.eye(4) # Transformation matrix
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T

def compute_theodolite_Transform(df_ground_control_points, reference_frame):
    df_ground_control_points['T'] = None
    
    P1 = []
    P2 = []
    P3 = []

    for index, row in df_ground_control_points.iterrows():
        id = int(row['id'])
        status = int(row['status'])
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])

        if status == 0:
            if id == 1:
                P1.append(np.array([X, Y, Z, 1]).T)
            if id == 2:
                P2.append(np.array([X, Y, Z, 1]).T)
            if id == 3:
                P3.append(np.array([X, Y, Z, 1]).T)

    P1 = np.array(P1).T
    P2 = np.array(P2).T
    P3 = np.array(P3).T

    if reference_frame == 1:
        T_I = np.identity(4)
        T_1_2 = np.identity(4)
        T_1_3 = np.identity(4)
        if id == 1:
            T_I = np.identity(4)
        if id == 2:
            T_1_2 = minimization(np.array(P2), np.array(P1))
            P2 = T_1_2 @ P2
        if id == 3:
            T_1_3 = minimization(np.array(P3), np.array(P1))
            P3 = T_1_3 @ P3
        return P1, P2, P3, T_I, T_1_2, T_1_3
        
    if reference_frame == 2:
        T_I = np.identity(4)
        T_2_1 = np.identity(4)
        T_2_3 = np.identity(4)
        if id == 2:
            T_I = np.identity(4)
        if id == 1:
            T_2_1 = minimization(np.array(P1), np.array(P2))
            P1 = T_2_1 @ P1
        if id == 3:
            T_2_3 = minimization(np.array(P3), np.array(P2))
            P3 = T_2_3 @ P3
        return P1, P2, P3, T_2_1, T_I, T_2_3
        
    if reference_frame == 3:
        T_I = np.identity(4)
        T_3_1 = np.identity(4)
        T_3_2 = np.identity(4)
        if id == 3:
            T_I = np.identity(4)
        if id == 1:
            T_3_1 = minimization(np.array(P1), np.array(P3))
            P1 = T_3_1 @ P1
        if id == 2:
            T_3_2 = minimization(np.array(P2), np.array(P3))
            P2 = T_3_2 @ P2
        return P1, P2, P3, T_3_1, T_3_2, T_I


def apply_theodolite_Transform_to_data(df, T_theodolite_1, T_from_theodolite_2_to_theodolite_1, T_from_theodolite_3_to_theodolite_1):
    df_theodolite_1 = df[df['id'] == 1]
    df_theodolite_2 = df[df['id'] == 2]
    df_theodolite_3 = df[df['id'] == 3]

    for index, row in df_theodolite_1.iterrows():
        timestamp = row['timestamp']
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])
        point = np.array([X, Y, Z, 1]).T
        point = T_theodolite_1 @ point
        df_theodolite_1.loc[index, 'X'] = point[0]
        df_theodolite_1.loc[index, 'Y'] = point[1]
        df_theodolite_1.loc[index, 'Z'] = point[2]
    
    for index, row in df_theodolite_2.iterrows():
        timestamp = row['timestamp']
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])
        point = np.array([X, Y, Z, 1]).T
        point = T_from_theodolite_2_to_theodolite_1 @ point
        df_theodolite_2.loc[index, 'X'] = point[0]
        df_theodolite_2.loc[index, 'Y'] = point[1]
        df_theodolite_2.loc[index, 'Z'] = point[2]
    
    for index, row in df_theodolite_3.iterrows():
        timestamp = row['timestamp']
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])
        point = np.array([X, Y, Z, 1]).T
        point = T_from_theodolite_3_to_theodolite_1 @ point
        df_theodolite_3.loc[index, 'X'] = point[0]
        df_theodolite_3.loc[index, 'Y'] = point[1]
        df_theodolite_3.loc[index, 'Z'] = point[2]

    df_ground_truth_theodolite_1 = pd.concat([df_theodolite_1, df_theodolite_2, df_theodolite_3]).sort_values('timestamp')
    return df_ground_truth_theodolite_1

def generate_ground_truth_lidar_frame(df_ground_truth_theodolite_1, df_calibration):
    # df_theodolite_frame = df_ground_truth_theodolite_1[['X', 'Y', 'Z']].to_numpy().T
    df_theodolite_frame = df_ground_truth_theodolite_1
    P_lidar_frame = df_calibration.to_numpy()[0:3, :].T

    df_theodolite_frame['group'] = (df_theodolite_frame.groupby('id')).apply(lambda x: pd.Series(np.arange(len(x)), x.index))
    print(df_theodolite_frame)
    # df_theodolite_frame.groupby('id' == group).count()

    # df_theodolite_frame['group'] = df_theodolite_frame.groupby('id').cumcount()
    # print(df_theodolite_frame['group'][0:50].to_markdown())
    # print(df_theodolite_frame[0:50].to_markdown())

    # blocks = df_theodolite_frame.groupby(['id', 'group']).filter(lambda x: len(x) == 3 and all(x['id'] == [1, 2, 3])).reset_index(drop=True)
    # print(blocks)

    # T_from_lidar_to_theodolite = minimization(P_lidar_frame, Q_theodolite_frame)
    # print(T_from_lidar_to_theodolite)
    return None
