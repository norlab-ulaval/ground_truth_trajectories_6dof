import numpy as np
import pandas as pd

#----------------- Data ----------------------
data = {
    'prism1' : {'ha_deg': 7, 'ha_min': 11, 'ha_sec': 12, 'va_deg': 90, 'va_min': 33, 'va_sec': 9, 'distance': 3.307},
    'prism2' : {'ha_deg': 0, 'ha_min': 00, 'ha_sec': 2, 'va_deg': 94, 'va_min': 41, 'va_sec': 43, 'distance': 4.021},
    'prism3' : {'ha_deg': 354, 'ha_min': 31, 'ha_sec': 52, 'va_deg': 96, 'va_min': 57, 'va_sec': 11, 'distance': 3.048},
    'lidar_top': {'ha_deg': 0, 'ha_min': 6, 'ha_sec': 0, 'va_deg': 101, 'va_min': 97, 'va_sec': 44, 'distance': 3.093},
    'lidar_left': {'ha_deg': 357, 'ha_min': 23, 'ha_sec': 16, 'va_deg': 103, 'va_min': 56, 'va_sec': 33, 'distance': 3.105},
    'lidar_right': {'ha_deg': 1, 'ha_min': 21, 'ha_sec': 26, 'va_deg': 104, 'va_min': 17, 'va_sec': 53, 'distance': 3.094}
}

df = pd.DataFrame(data).T

#----------------- Constants ----------------------
DISTANCE_LIDAR_TOP_TO_LIDAR_ORIGIN = 0.063  # meters, for RS32 on warthog

#----------------- Calibration Functions ----------------------
def compute_transformation_to_lidar_frame(df_calibration, distance_lidar_top_to_lidar_origin, debug=False):
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
    return df_calibration, T_from_theodolite_to_lidar

def angles_to_coordinates(df):
    df['ha'] = df['ha_deg'] + df['ha_min'] / 60 + df['ha_sec'] / 3600
    df['va'] = df['va_deg'] + df['va_min'] / 60 + df['va_sec'] / 3600
    df['distance'] = df['distance']
    return df

def data_to_3D_points_deg_to_rad_calibration(df):
    df['X'] = df['distance'] * np.cos(-df['ha'] * np.pi / 180) * np.cos((90 - df['va']) * np.pi / 180)
    df['Y'] = df['distance'] * np.sin(-df['ha'] * np.pi / 180) * np.cos((90 - df['va']) * np.pi / 180)
    df['Z'] = df['distance'] * np.sin((90 - df['va']) * np.pi / 180)
    return df

def visualize(df):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'])
    for index, row in df.iterrows():
        ax.text(row['X'] - 0.02, row['Y'] + 0.02, row['Z'] + 0.02, index)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sensors in Lidar frame')
    plt.show()

df = angles_to_coordinates(df)
df = data_to_3D_points_deg_to_rad_calibration(df)
df_calibration = df.drop(columns=['ha_deg', 'ha_min', 'ha_sec', 'va_deg', 'va_min', 'va_sec', 'ha', 'va', 'distance'])
df_calibration, T_from_theodolite_to_lidar = compute_transformation_to_lidar_frame(df_calibration, DISTANCE_LIDAR_TOP_TO_LIDAR_ORIGIN, False)
print('\nSensors positions in Lidar frame:\n', df_calibration, '\n')
print('Theodolite frame to Lidar frame Transformation matrix:\n', T_from_theodolite_to_lidar)
visualize(df_calibration)
