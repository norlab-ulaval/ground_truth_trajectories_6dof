import numpy as np
import math

def coordinates_transformation(d, ha, va, param):
    if param == 'deg':
        x = d * math.cos((-ha) * np.pi / 180) * math.cos((90 - va) * np.pi / 180)
        y = d * math.sin((-ha) * np.pi / 180) * math.cos((90 - va) * np.pi / 180)
        z = d * math.sin((90 - va) * np.pi / 180)
    if param == 'rad':
        x = d * math.cos(-ha) * math.cos(np.pi / 2 - va)
        y = d * math.sin(-ha) * math.cos(np.pi / 2 - va)
        z = d * math.sin(np.pi / 2 - va)
    return np.array([x, y, z, 1], dtype=np.float64)

def read_dist_inter_prism_gps(file_path, file_name_output, name_lidar):
    with open(file_path, "r") as file:
        lines = file.readlines()

    points = []
    for line in lines:
        item = line.split(" ")
        ha = float(item[0]) + float(item[1]) / 60 + float(item[2]) / 3600
        va = float(item[6]) + float(item[7]) / 60 + float(item[8]) / 3600
        d = float(item[12])
        points.append(coordinates_transformation(d, ha, va, 'deg'))

    dp12 = np.linalg.norm(points[0] - points[1], axis=0)
    dp13 = np.linalg.norm(points[0] - points[2], axis=0)
    dp23 = np.linalg.norm(points[1] - points[2], axis=0)
    print('\n Distance inter-prism [m] :')
    print("1 - 2 :", dp12)
    print("1 - 3 :", dp13)
    print("2 - 3 :", dp23)

    if len(points) > 3:
        dg12 = np.linalg.norm(points[3] - points[4], axis=0)
        dg13 = np.linalg.norm(points[3] - points[5], axis=0)
        dg23 = np.linalg.norm(points[4] - points[5], axis=0)
        print('\n Distance inter-GPS [m] :')
        print("1 - 2 :", dg12)
        print("1 - 3 :", dg13)
        print("2 - 3 :", dg23)

        if len(points) > 6 and name_lidar == "Robosense_32":
            distance_lidar_top_to_lidar_origin = 0.063  # In meter, for RS32 on warthog
            L1, L2, L3 = points[6:9]

            u = L2[:3] - L1[:3]
            v = L3[:3] - L1[:3]
            A = np.array([[u[0], u[1]], [v[0], v[1]]])
            y = np.array([-u[2], -v[2]])
            x = np.linalg.solve(A, y)
            n = np.array([x[0], x[1], 1])

            I = L2[:3] + np.dot(L1[:3] - L2[:3], L3[:3] - L2[:3]) / np.dot(L3[:3] - L2[:3], L3[:3] - L2[:3]) * (L3[:3] - L2[:3])
            z = I - L1[:3]
            z_unit = z / np.linalg.norm(z)
            O = distance_lidar_top_to_lidar_origin * z_unit + L1[:3]
            Ox = -1 / np.linalg.norm(n) * n + O
            Oz = -1 * z_unit + O
            Oy = np.cross(-z_unit, -1 / np.linalg.norm(n) * n) + O

        with open(file_name_output, "w+") as csv_file:
            csv_file.write(f"{dp12} {dp13} {dp23}")
            if len(points) > 3:
                csv_file.write(f" {dg12} {dg13} {dg23}")
            csv_file.write("\n")

def read_sensor_positions(file_path, file_name_output, name_lidar):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            item = line.split(" ")
            ha = float(item[0]) + float(item[1]) * 1 / 60 + float(item[2]) * 1 / 3600
            va = float(item[6]) + float(item[7]) * 1 / 60 + float(item[8]) * 1 / 3600
            d = float(item[12])
            points.append(coordinates_transformation(d, ha, va, 'deg'))

    P1, P2, P3, G1, G2, G3 = points[:6]

    if len(points) > 6 and name_lidar == "Robosense_32":
        distance_lidar_top_to_lidar_origin = 0.063  # In meter, for RS32 on warthog
        L1, L2, L3 = points[6:9]
        u = L2[:3] - L1[:3]
        v = L3[:3] - L1[:3]
        A = np.array([[u[0], u[1]], [v[0], v[1]]])
        y = np.array([-u[2], -v[2]])
        x = np.linalg.solve(A, y)
        n = np.array([x[0], x[1], 1])
        I = L2[:3] + np.dot(L1[:3] - L2[:3], L3[:3] - L2[:3]) / np.dot(L3[:3] - L2[:3], L3[:3] - L2[:3]) * (L3[:3] - L2[:3])
        z = I - L1[:3]
        z_unit = 1 / np.linalg.norm(z) * z
        O = distance_lidar_top_to_lidar_origin * z_unit + L1[:3]
        Ox = -1 / np.linalg.norm(n) * n + O
        Oz = -1 * z_unit + O
        Oy = 1 * np.cross(-z_unit, -1 / np.linalg.norm(n) * n) + O

    with open(file_name_output, "w+") as csv_file:
        csv_file.write(f"{P1[0]} {P1[1]} {P1[2]} 1\n")
        csv_file.write(f"{P2[0]} {P2[1]} {P2[2]} 1\n")
        csv_file.write(f"{P3[0]} {P3[1]} {P3[2]} 1\n")
        csv_file.write(f"{G1[0]} {G1[1]} {G1[2]} 1\n")
        csv_file.write(f"{G2[0]} {G2[1]} {G2[2]} 1\n")
        csv_file.write(f"{G3[0]} {G3[1]} {G3[2]} 1\n")
        if len(points) > 6 and name_lidar == "Robosense_32":
            csv_file.write(f"{O[0]} {O[1]} {O[2]} 1\n")
            csv_file.write(f"{Ox[0]} {Ox[1]} {Ox[2]} 1\n")
            csv_file.write(f"{Oy[0]} {Oy[1]} {Oy[2]} 1\n")
            csv_file.write(f"{Oz[0]} {Oz[1]} {Oz[2]} 1\n")
    print('\nSensors coordinates in theodolite frame :')
    print(f"Prism 1: {P1}")
    print(f"Prism 2: {P2}")
    print(f"Prism 3: {P3}")
    print(f"GNSS 1: {G1}")
    print(f"GNSS 2: {G2}")
    print(f"GNSS 3: {G3}")
    if len(points) > 6 and name_lidar == "Robosense_32":
        print('\nLiDAR center :')
        print(f"O: {O}")
        print(f"Ox: {Ox}")
        print(f"Oy: {Oy}")
        print(f"Oz: {Oz}")

def read_file(path_file):
    if path_file == "":
        return []
    else:
        value = np.genfromtxt(path_file, delimiter=" ")
        return value

def sensors_positions_lidar_frame(sensors, file_name_output):
    P1, P2, P3, G1, G2, G3, O, Ox, Oy, Oz = sensors[:10]
    
    nx = Ox - O
    ny = Oy - O
    nz = Oz - O

    T_lidar = np.array([[nx[0],ny[0],nz[0],O[0]],
                        [nx[1],ny[1],nz[1],O[1]],
                        [nx[2],ny[2],nz[2],O[2]],
                        [0,0,0,1]])
    
    T_lidar_inv = np.linalg.inv(T_lidar)
    print('Transformation matrix of theodolite to LiDAR frame :\n', T_lidar_inv)

    P1_lidar = T_lidar_inv @ P1
    P2_lidar = T_lidar_inv @ P2
    P3_lidar = T_lidar_inv @ P3
    GPS1_lidar = T_lidar_inv @ G1
    GPS2_lidar = T_lidar_inv @ G2
    GPS3_lidar = T_lidar_inv @ G3
    O_lidar = T_lidar_inv @ O

    with open(file_name_output, "w+") as csv_file:
        csv_file.write(f"{P1_lidar[0]} {P1_lidar[1]} {P1_lidar[2]} 1\n")
        csv_file.write(f"{P2_lidar[0]} {P2_lidar[1]} {P2_lidar[2]} 1\n")
        csv_file.write(f"{P3_lidar[0]} {P3_lidar[1]} {P3_lidar[2]} 1\n")
        csv_file.write(f"{GPS1_lidar[0]} {GPS1_lidar[1]} {GPS1_lidar[2]} 1\n")
        csv_file.write(f"{GPS2_lidar[0]} {GPS2_lidar[1]} {GPS2_lidar[2]} 1\n")
        csv_file.write(f"{GPS3_lidar[0]} {GPS3_lidar[1]} {GPS3_lidar[2]} 1\n")
        csv_file.write(f"{O_lidar[0]} {O_lidar[1]} {O_lidar[2]} 1\n")

    print('\nSensors coordinates in LiDAR frame :')
    print(f"Prism 1: {P1_lidar}")
    print(f"Prism 2: {P2_lidar}")
    print(f"Prism 3: {P3_lidar}")
    print(f"GNSS 1: {GPS1_lidar}")
    print(f"GNSS 2: {GPS2_lidar}")
    print(f"GNSS 3: {GPS3_lidar}")
    print(f"GNSS 3: {O_lidar}")