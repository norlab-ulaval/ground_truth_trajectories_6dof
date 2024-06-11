import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def display_ground_control_points(df_GCP, title):
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(df_GCP['X'], df_GCP['Y'], color = "C1")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

def display_ground_truth(df):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(df['X'], df['Y'], df['Z'], color = "C1")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ground truth data')