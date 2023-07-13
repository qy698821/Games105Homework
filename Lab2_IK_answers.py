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
    cycle_limit = 50
    target_joint_id = path1_index_list[0]
    error_check = 0.01
    parent_index_list = meta_data.joint_parent

    original_positions = np.copy(joint_positions)
    original_orientations = np.copy(joint_orientations)

    # CCD
    for i in range(cycle_limit):
        # path1
        for index in range(len(path1_index_list)):
            if index == 0:
                continue
            joint_index = path1_index_list[index]
            joint_vector = joint_positions[target_joint_id] - joint_positions[joint_index]
            target_vector = target_pose - joint_positions[joint_index]
            matrix = rotation_matrix_from_vectors(joint_vector, target_vector)
            joint_orientation = R.from_quat(joint_orientations[joint_index]).as_matrix()
            matrix_format = R.from_matrix(matrix).as_matrix()
            joint_orientations[joint_index] = R.from_matrix(matrix_format.dot(joint_orientation)).as_quat()

            # calculate the children joints
            child_i = index - 1
            while child_i >= 0:
                child_joint_id = path1_index_list[child_i]
                if child_i != 0:
                    # children orientation calculate
                    child_orientation = R.from_quat(joint_orientations[child_joint_id]).as_matrix()
                    joint_orientations[child_joint_id] = R.from_matrix(matrix_format.dot(child_orientation)).as_quat()

                # Children position calculate
                child_vector = joint_positions[child_joint_id] - joint_positions[joint_index]
                # rotate the vector
                child_vector_new = matrix.dot(child_vector)
                joint_positions[child_joint_id] = child_vector_new + joint_positions[joint_index]
                child_i -= 1
        # # path2
        # for index in range(len(path2_index_list)-1,0,-1):
        #     joint_index = path2_index_list[index]
        #     joint_vector = joint_positions[target_joint_id] - joint_positions[joint_index]
        #     target_vector = target_pose - joint_positions[joint_index]
        #     matrix = rotation_matrix_from_vectors(joint_vector, target_vector)
        #     joint_orientation = R.from_quat(joint_orientations[joint_index]).as_matrix()
        #     matrix_format = R.from_matrix(matrix).as_matrix()
        #     joint_orientations[joint_index] = R.from_matrix(matrix_format.dot(joint_orientation)).as_quat()
        #
        #     # calculate the children joints
        #     child_i = index - 1
        #     while child_i >= 0:
        #         child_joint_id = path2_index_list[child_i]
        #         if child_i != 0:
        #             # children orientation calculate
        #             child_orientation = R.from_quat(joint_orientations[child_joint_id]).as_matrix()
        #             joint_orientations[child_joint_id] = R.from_matrix(matrix_format.dot(child_orientation)).as_quat()
        #
        #         # Children position calculate
        #         child_vector = joint_positions[child_joint_id] - joint_positions[joint_index]
        #         # rotate the vector
        #         child_vector_new = matrix.dot(child_vector)
        #         joint_positions[child_joint_id] = child_vector_new + joint_positions[joint_index]
        #         child_i -= 1

        current_error = np.linalg.norm(joint_positions[target_joint_id] - target_pose)
        if current_error < error_check:
            print("total calculate : ", i, )
            break
    # calculate other joint rotation and position
    for i in range(len(joint_orientations)):
        if i in path_index_list:
            continue
        joint_original_rot_matrix = R.from_quat(original_orientations[i]).as_matrix()
        joint_local_rot = np.linalg.inv(R.from_quat(original_orientations[parent_index_list[i]]).as_matrix()) * joint_original_rot_matrix
        parent_rot_matrix = R.from_quat(joint_orientations[parent_index_list[i]]).as_matrix()
        new_rot_matrix = parent_rot_matrix.dot(joint_local_rot)
        joint_orientations[i] = R.from_matrix(new_rot_matrix).as_quat()


        parent_original_rot_matrix = R.from_quat(original_orientations[parent_index_list[i]]).as_matrix()
        parent_original_position = original_positions[parent_index_list[i]]
        joint_original_position = original_positions[i]
        joint_local_position = joint_original_position - parent_original_position
        delta_orientation = np.dot(new_rot_matrix, np.linalg.inv(parent_original_rot_matrix))
        joint_positions[i] = joint_positions[parent_index_list[i]] + delta_orientation.dot(joint_local_position)


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