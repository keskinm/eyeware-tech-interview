import os
import random
import numpy as np
from numpy import linalg as LA
import shutil
import re

def create_files():
    work_dir = './data/task1'
    files_n = 20
    vectors_file_path = os.path.join(work_dir, 'vectors.txt')
    with open(vectors_file_path, "w") as opened_vectors_file:
        for i in range(files_n):
            random_coords = [random.randrange(-10, 10) for j in range(3)]
            opened_vectors_file.write(
                "file_{i}.data\t{vx}\t{vy}\t{vz}\n".format(i=i, vx=random_coords[0], vy=random_coords[1],
                                                           vz=random_coords[2]))

    for i in range(files_n):
        data_file_path = os.path.join(work_dir, 'file_{}.data'.format(i))
        with open(data_file_path, "w+") as opened_data_file:
            opened_data_file.write('')


create_files()

def parse_vectors():
    work_dir = './data/task1'
    vectors_file_path = os.path.join(work_dir, 'vectors.txt')

    with open(vectors_file_path, "r") as opened_vectors_file:
        # content = opened_vectors_file.readlines()
    # content = [x.strip() for x in content]
    # print(content)

        lines = []
        for line in opened_vectors_file:
            line = line.strip().split()[1:]
            line = [int(i) for i in line]
            lines.append(line)
        data = np.array(lines)
        main_vector = (data.mean(axis=0))

    return data, main_vector

data, main_vector = parse_vectors()

def sort_files(data, main_vector):
    files_n = 20
    index = np.expand_dims(np.arange(files_n), axis=1)
    indexed_data = np.concatenate([index, data], axis=1)
    dist = LA.norm(data-main_vector, axis=1)

    sorted_eucl_dist = np.argsort(dist)
    folder_a_end_idx = round(0.4*files_n)
    folder_a_idx, folder_a_compl_idx = sorted_eucl_dist[0:folder_a_end_idx], sorted_eucl_dist[folder_a_end_idx:]
    folder_a_data, folder_a_compl_data = data[folder_a_idx], data[folder_a_compl_idx]

    folder_b_idx, folder_c_idx = folder_a_compl_idx[np.where(folder_a_compl_data[:, 2] >= 0)], folder_a_compl_idx[np.where(folder_a_compl_data[:, 2] < 0)]

    return folder_a_idx, folder_b_idx, folder_c_idx


folder_a_idx, folder_b_idx, folder_c_idx = sort_files(data, main_vector)


def move_files(folder_a_idx, folder_b_idx, folder_c_idx):
    work_dir = './data/task1'


    folder_a_dir_path = './data/task1/folder_A'
    folder_b_dir_path = './data/task1/folder_B'
    folder_c_dir_path = './data/task1/folder_C'

    shutil.rmtree(folder_a_dir_path)
    shutil.rmtree(folder_b_dir_path)
    shutil.rmtree(folder_c_dir_path)


    file_names = os.listdir(work_dir)

    os.makedirs(folder_a_dir_path, exist_ok=True)
    os.makedirs(folder_b_dir_path, exist_ok=True)
    os.makedirs(folder_c_dir_path, exist_ok=True)

    for file_name in file_names:
        if not 'vectors' in file_name:
            file_idx_with_extension = file_name.split('_')[1]
            file_idx = os.path.splitext(file_idx_with_extension)[0]
            file_path = os.path.join(work_dir, file_name)

            print(((file_idx)))

            if int(file_idx) in folder_a_idx:
                shutil.move(file_path, os.path.join(folder_a_dir_path, file_name))

            elif int(file_idx) in folder_b_idx:
                shutil.move(file_path, os.path.join(folder_b_dir_path, file_name))

            elif int(file_idx) in folder_c_idx:
                shutil.move(file_path, os.path.join(folder_c_dir_path, file_name))



move_files(folder_a_idx, folder_b_idx, folder_c_idx)
