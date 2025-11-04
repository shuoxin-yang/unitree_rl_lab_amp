import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import torch


class Motion:
    def __init__(self):
        self.action_pairs = []  # List of (state, next_state) pairs

    def load_motions(
        self,
        motion_folder: str,
        motion_files: list[str],
        weights: list[float],
        target_fps: int,
    ):
        """
        加载运动数据组
        参数：
            motion_folder: 运动数据文件夹路径
            motion_files: 运动数据文件名列表（不含扩展名）
            weights: 每个运动数据的权重列表
        """
        for motion_file, weight in zip(motion_files, weights):
            motion_data = f"{motion_folder}/{motion_file}.npy"
            self.load_motion(motion_data, target_fps)

    def load_motion(self, motion_file: str, target_fps):
        """
        加载单个运动数据文件,进行插值和计算，并保存到action_pairs中
        参数：
            motion_file: 运动数据文件路径(含扩展名)
        """
        data = np.load(motion_file, allow_pickle=True).item()
        original_dof_positions = data["dof_pos"]
        original_root_positions = data["root_pos"]
        original_root_rotation = data["root_rot"]
        original_fps = data["fps"]
        itp_dof_pos, itp_dof_vel, itp_root_lin_vel, itp_root_ang_vel = (
            self.interpolate_motion_data(
                original_dof_positions,
                original_root_positions,
                original_root_rotation,
                original_fps,
                target_fps,
            )
        )
        for i in range(itp_dof_pos.shape[0] - 1):
            state = torch.cat(
                [
                    itp_dof_pos[i],
                    itp_dof_vel[i],
                    itp_root_lin_vel[i],
                    itp_root_ang_vel[i],
                ],
                dim=0,
            )
            next_state = torch.cat(
                [
                    itp_dof_pos[i + 1],
                    itp_dof_vel[i + 1],
                    itp_root_lin_vel[i + 1],
                    itp_root_ang_vel[i + 1],
                ],
                dim=0,
            )
            state_trans_pair = torch.cat([state, next_state], dim=0)
            self.action_pairs.append(state_trans_pair)

    def interpolate_motion_data(self, dof_pos, root_pos, root_rot, fps, target_fps):
        """
        对动作捕捉数据进行线性插值并计算速度/角速度
        
        参数:
        dof_pos: numpy.ndarray, shape=(num_frames, 29) - 关节自由度位置数据
        root_pos: numpy.ndarray, shape=(num_frames, 3) - 根节点位置数据
        root_rot: numpy.ndarray, shape=(num_frames, 4) - 根节点旋转数据(四元数)
        fps: int - 原始帧率
        target_fps: int - 目标帧率
        
        返回:
        dict: 包含插值后的位置、速度、角速度等数据的字典
        """
        
        num_frames = dof_pos.shape[0]
        
        # 计算时间轴
        original_time = np.arange(num_frames) / fps
        target_time = np.arange(0, original_time[-1], 1/target_fps)
        num_target_frames = len(target_time)
        
        # ------------------------------
        # DOF位置插值和速度计算
        # ------------------------------
        # 使用scipy的interp1d进行线性插值
        dof_interpolator = interp1d(original_time, dof_pos, axis=0, kind='linear', fill_value='extrapolate')
        dof_pos_interpolated = dof_interpolator(target_time)
        
        # 计算速度 (单位: 单位/秒)
        dof_vel = np.zeros_like(dof_pos_interpolated)
        dof_vel[1:] = (dof_pos_interpolated[1:] - dof_pos_interpolated[:-1]) * target_fps
        
        # ------------------------------
        # 根节点位置插值和速度计算
        # ------------------------------
        # 使用scipy的interp1d进行线性插值
        root_pos_interpolator = interp1d(original_time, root_pos, axis=0, kind='linear', fill_value='extrapolate')
        root_pos_interpolated = root_pos_interpolator(target_time)
        
        # 计算速度 (单位: 单位/秒)
        root_vel = np.zeros_like(root_pos_interpolated)
        root_vel[1:] = (root_pos_interpolated[1:] - root_pos_interpolated[:-1]) * target_fps
        
        # ------------------------------
        # 根节点旋转插值和角速度计算
        # ------------------------------
        # 使用scipy的Slerp进行球面线性插值
        rotations = Rotation.from_quat(root_rot)
        slerp = Slerp(original_time, rotations)
        rotations_interpolated = slerp(target_time)
        root_rot_interpolated = rotations_interpolated.as_quat()
        
        # 计算角速度 (单位: 弧度/秒)
        root_angular_vel = np.zeros_like(root_rot_interpolated)
        
        for i in range(1, num_target_frames):
            # 计算旋转增量
            delta_rot = rotations_interpolated[i] * rotations_interpolated[i-1].inv()
            
            # 转换为角速度向量
            angle = delta_rot.magnitude()
            if angle < 1e-6:
                axis = np.array([1, 0, 0])  # 默认轴
            else:
                axis = delta_rot.as_rotvec() / angle
            
            # 计算角速度 (弧度/秒)
            root_angular_vel[i] = np.concatenate([[0], axis * angle * target_fps])
        
        # 转换为PyTorch张量
        return torch.tensor(dof_pos_interpolated, dtype=torch.float32), torch.tensor(dof_vel, dtype=torch.float32), torch.tensor(root_vel, dtype=torch.float32), torch.tensor(root_angular_vel, dtype=torch.float32)
