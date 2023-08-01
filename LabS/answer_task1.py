from bvh_utils import *
#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    zero_list = [0, 0, 0]

    #---------------你的代码------------------#
    for i in range(len(skinning_weight)):
        T_translation = T_pose_vertex_translation[i]
        new_translation = np.array(zero_list)
        for j in range(len(skinning_weight[i])):
            if skinning_weight[i][j] != 0:
                current_joint_index = skinning_idx[i][j]
                offset = T_translation - T_pose_joint_translation[current_joint_index]
                joint_change = (R.from_quat(joint_orientation[current_joint_index]).apply(offset) + joint_translation[current_joint_index]) * skinning_weight[i][j]
                new_translation = new_translation + joint_change
        vertex_translation[i] = new_translation

    
    return vertex_translation