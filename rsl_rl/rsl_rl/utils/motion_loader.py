import numpy as np
from scipy.spatial.transform import Rotation as R


class MotionLoader:
    def __init__(self, motion_file: str, target_fps: float):
        """
        初始化运动加载器，加载运动数据并插值到目标帧率
        参数：
            motion_file: 运动数据文件路径（.npy格式）
            target_fps: 目标帧率（如50fps）
        """
        # 加载运动数据并校验必要键
        self.motion_data = np.load(motion_file, allow_pickle=True).item()
        required_keys = ["joint_positions", "root_positions", "root_orientations", "fps"]
        for key in required_keys:
            if key not in self.motion_data:
                raise ValueError(f"运动数据缺少必要键: {key}")
        
        # 原始数据参数
        self.original_fps = self.motion_data["fps"]
        self.original_num_frames = self.motion_data["joint_positions"].shape[0]
        self.target_fps = target_fps
        
        # 计算原始时间轴和目标时间轴（单位：秒）
        self.original_time = np.linspace(
            0, 
            (self.original_num_frames - 1) / self.original_fps,  # 总时长
            self.original_num_frames
        )
        self.target_num_frames = int(self.original_time[-1] * self.target_fps) + 1  # 目标帧数
        self.target_time = np.linspace(
            0, 
            self.original_time[-1],  # 保持总时长不变
            self.target_num_frames
        )
        
        # 插值核心数据
        self.interp_joint_positions = self._interpolate_positions(
            self.motion_data["joint_positions"]
        )
        self.interp_root_positions = self._interpolate_positions(
            self.motion_data["root_positions"]
        )
        self.interp_root_orientations = self._interpolate_orientations(
            self.motion_data["root_orientations"]
        )
        
        # 重新计算速度（基于插值后的数据和目标帧率）
        self.joint_velocities = self.compute_velocities(
            self.interp_joint_positions, self.target_fps
        )
        self.root_velocities = self.compute_velocities(
            self.interp_root_positions, self.target_fps
        )

    def _interpolate_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        对位置数据（关节位置/根位置）进行线性插值
        参数：
            positions: 原始位置数据，形状为 (original_num_frames, ..., 3)
                       例如关节位置：(N, num_joints, 3)；根位置：(N, 3)
        返回：
            插值后的位置数据，形状为 (target_num_frames, ..., 3)
        """
        # 处理任意维度的位置数据（只要第一维是时间帧）
        shape = positions.shape
        num_dims = len(shape)
        interp_pos = np.zeros((self.target_num_frames, *shape[1:]))
        
        # 对每个空间维度单独插值
        for dim in range(3):
            # 展平除时间和空间维度外的其他维度（如关节索引）
            if num_dims == 2:  # 根位置：(N, 3)
                interp_pos[..., dim] = np.interp(
                    self.target_time, self.original_time, positions[..., dim]
                )
            elif num_dims == 3:  # 关节位置：(N, num_joints, 3)
                for joint in range(shape[1]):
                    interp_pos[:, joint, dim] = np.interp(
                        self.target_time, self.original_time, positions[:, joint, dim]
                    )
            else:
                raise ValueError(f"不支持的位置数据维度: {num_dims}")
        return interp_pos

    def _interpolate_orientations(self, orientations: np.ndarray) -> np.ndarray:
        """
        对根朝向四元数进行球面线性插值（SLERP），保证旋转连贯性
        参数：
            orientations: 原始四元数数据，形状为 (original_num_frames, 4)（w, x, y, z 或 x, y, z, w，需与scipy兼容）
        返回：
            插值后的四元数数据，形状为 (target_num_frames, 4)
        """
        # 转换为scipy的Rotation对象（自动处理四元数格式）
        original_rots = R.from_quat(orientations)
        # 计算插值比例（0到1之间，对应目标时间在原始时间中的位置）
        t = self.target_time / self.original_time[-1] if self.original_time[-1] != 0 else 0
        # 球面线性插值
        interp_rots = original_rots.slerp(t)
        return interp_rots.as_quat()

    def compute_velocities(self, positions: np.ndarray, fps: float) -> np.ndarray:
        """
        根据位置数据计算速度（单位：单位/秒）
        参数：
            positions: 位置数据，形状为 (num_frames, ..., 3)
            fps: 帧率，用于计算时间间隔（1/fps）
        返回：
            速度数据，形状与positions一致
        """
        num_frames = positions.shape[0]
        velocities = np.zeros_like(positions)
        if num_frames < 2:
            return velocities  # 帧数不足时速度为0
        
        # 前向差分计算速度（v = Δpos / Δt = (pos[i+1] - pos[i]) * fps）
        velocities[:-1] = (positions[1:] - positions[:-1]) * fps
        # 最后一帧复用前一帧的速度（避免越界）
        velocities[-1] = velocities[-2]
        return velocities

    def get_interpolated_data(self) -> dict:
        """返回插值后的完整运动数据"""
        return {
            "joint_positions": self.interp_joint_positions,
            "root_positions": self.interp_root_positions,
            "root_orientations": self.interp_root_orientations,
            "joint_velocities": self.joint_velocities,
            "root_velocities": self.root_velocities,
            "fps": self.target_fps,
            "num_frames": self.target_num_frames
        }
