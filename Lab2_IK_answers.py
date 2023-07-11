import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix_from_vectors(vector1, vector2):
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)

    rotation_axis = np.cross(vector1_norm, vector2_norm)
    rotation_angle = np.arccos(np.dot(vector1_norm, vector2_norm))

    rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)
    return rotation_matrix

def rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    rotation_matrix = np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis, axis) + sin_theta * K
    return rotation_matrix


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path_index_list, path_name_list, path1_index_list, path2_index_list = meta_data.get_path_from_root_to_end()
    cycle_limit = 20
    for i in range(cycle_limit):
        for index in range(len(path1_index_list)):
            if index == len(path1_index_list) - 1:
                break
            joint_vector = joint_positions[path1_index_list[index]] - joint_positions[path1_index_list[index + 1]]
            target_vector = target_pose - joint_positions[path1_index_list[index + 1]]
            matrix1 = rotation_matrix(np.array([1,0,0]),np.array([1,1,0]))
            matrix2 = rotation_matrix_from_vectors(np.array([1,0,0]),np.array([1,1,0]))




    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations