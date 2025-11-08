import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import torch
import os


class Motion:
    def __init__(self, default_joint_pos: list[float], train_eval_rate: float, device: str = "cuda:0"):
        """
        参数：
            default_joint_pos: 机器人默认关节位置，若运动数据包含默认位置，则需要传入此参数，否则请传入[0,...]
            train_eval_rate: 样本训练评估比，若传入0.8，则80%的样本将作为训练样本，20%的样本作为评估样本，观测过拟合问题
        """
        self.device = device
        self.train_action_pairs = []  # List of (state, next_state) pairs
        self.eval_action_pairs = []
        self.train_eval_rate = train_eval_rate
        self.default_joint_pos = default_joint_pos

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
        注意：
            关键字"root_rot"的顺序应为xyzw，以符合scipy的rotation转换函数
        """
        if motion_files[0] == "*":
            from pathlib import Path

            npy_dir = Path(motion_folder)
            motion_files = [
                file.stem for file in npy_dir.glob("*.npy") if file.is_file()
            ]
            weights = np.ones_like(motion_files).tolist()
        print(f"[INF_MotionLoader]: Total files: {len(motion_files)}")
        for motion_file, weight in zip(motion_files, weights):
            for _ in range(int(weight)):
                self.load_motion(motion_folder, motion_file, target_fps)
                print(f",\tweight: {weight}")
        print(
            f"[INF_MotionLoader]: Total state pairs: {len(self.eval_action_pairs)+len(self.train_action_pairs)}"
        )

    def load_motion(self, motion_path: str, file_name: str, target_fps):
        """
        加载单个运动数据文件,进行插值和计算，并保存到action_pairs中
        参数：
            motion_path: 运动数据文件路径(含扩展名)
            file_name: 运动数据文件名
        注意：
            关键字"root_rot"的顺序应为xyzw，以符合scipy的rotation转换函数
        """
        motion_file = f"{motion_path}/{file_name}.npy"
        data = np.load(motion_file, allow_pickle=True).item()
        original_dof_positions = data["dof_pos"] - self.default_joint_pos
        original_root_positions = data["root_pos"]
        original_root_rotation = data["root_rot"]
        original_fps = data["fps"]
        (
            itp_dof_pos,
            itp_dof_vel,
            itp_root_lin_vel,
            itp_root_ang_vel,
            itp_root_pos,
            itp_root_rot,
        ) = self.interpolate_motion_data(
            original_dof_positions,
            original_root_positions,
            original_root_rotation,
            original_fps,
            target_fps,
        )
        npy_file = os.path.join(f"{motion_path}/interpolate/{file_name}.npy")
        npy = {
            "dof_pos": itp_dof_pos.numpy(),
            "root_pos": itp_root_pos.numpy(),
            "root_rot": itp_root_rot.numpy(),
            "fps": target_fps,
        }
        np.save(npy_file, npy)
        print(
            f"[INF_MotionLoader]: File name: {file_name},\tInitFPS: {original_fps:.4f},\tTotal frames: {original_dof_positions.shape[0]}",
            end="",
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
            ).to(self.device)
            next_state = torch.cat(
                [
                    itp_dof_pos[i + 1],
                    itp_dof_vel[i + 1],
                    itp_root_lin_vel[i + 1],
                    itp_root_ang_vel[i + 1],
                ],
                dim=0,
            ).to(self.device)
            state_trans_pair = torch.cat([state, next_state], dim=0).to(self.device)
            if torch.rand((1)) < self.train_eval_rate:
                self.train_action_pairs.append(state_trans_pair)
            else:
                self.eval_action_pairs.append(state_trans_pair)

    def random_get_train_action_pair(self):
        """随机获取一个训练动作对"""
        index = np.random.randint(0, len(self.train_action_pairs), 1)
        return self.train_action_pairs[index[0]]
    
    def random_get_eval_action_pair(self):
        """随机获取一个评估动作对"""
        index = np.random.randint(0, len(self.eval_action_pairs), 1)
        return self.eval_action_pairs[index[0]]

    def random_get_train_action_pair_batch(self, batch_size: int):
        """随机获取一批训练动作对"""
        indices = np.random.randint(0, len(self.train_action_pairs), batch_size)
        batch = torch.stack([self.train_action_pairs[i] for i in indices], dim=0)
        return batch
    
    def random_get_eval_action_pair_batch(self, batch_size: int):
        """随机获取一批评估动作对"""
        if batch_size > len(self.eval_action_pairs):
            print("[INF_MotionLoader] Too large size for eval pairs")
            return torch.tensor([], device=self.device)
        indices = np.random.randint(0, len(self.eval_action_pairs), batch_size)
        batch = torch.stack([self.eval_action_pairs[i] for i in indices], dim=0)
        return batch

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
        target_time = np.arange(0, original_time[-1], 1 / target_fps)
        num_target_frames = len(target_time)

        # ------------------------------
        # DOF位置插值和速度计算
        # ------------------------------
        # 使用scipy的interp1d进行线性插值
        dof_interpolator = interp1d(
            original_time, dof_pos, axis=0, kind="linear", fill_value="extrapolate"
        )
        dof_pos_interpolated = dof_interpolator(target_time)

        # 计算速度 (单位: 单位/秒)
        dof_vel = np.zeros_like(dof_pos_interpolated)
        dof_vel[1:] = (
            dof_pos_interpolated[1:] - dof_pos_interpolated[:-1]
        ) * target_fps

        # ------------------------------
        # 根节点位置插值和速度计算
        # ------------------------------
        # 使用scipy的interp1d进行线性插值
        root_pos_interpolator = interp1d(
            original_time, root_pos, axis=0, kind="linear", fill_value="extrapolate"
        )
        root_pos_interpolated = root_pos_interpolator(target_time)

        # 计算速度 (单位: 单位/秒)
        root_vel = np.zeros_like(root_pos_interpolated)
        root_vel[1:] = (
            root_pos_interpolated[1:] - root_pos_interpolated[:-1]
        ) * target_fps

        # ------------------------------
        # 根节点旋转插值和角速度计算
        # ------------------------------
        # 使用scipy的Slerp进行球面线性插值
        rotations = Rotation.from_quat(root_rot)
        slerp = Slerp(original_time, rotations)
        rotations_interpolated = slerp(target_time)

        # 计算角速度 (单位: 弧度/秒)
        root_angular_vel = np.zeros(shape=(num_target_frames, 3), dtype=np.float64)

        for i in range(1, num_target_frames):
            # 计算旋转增量
            delta_rot = rotations_interpolated[i] * rotations_interpolated[i - 1].inv()

            # 转换为角速度向量
            angle = delta_rot.magnitude()
            if angle < 1e-6:
                axis = np.array([1, 0, 0])  # 默认轴
            else:
                axis = delta_rot.as_rotvec() / angle

            # 计算角速度 (弧度/秒)
            root_angular_vel[i] = axis * angle * target_fps

        # 转换为PyTorch张量
        return (
            torch.tensor(dof_pos_interpolated, dtype=torch.float32),
            torch.tensor(dof_vel, dtype=torch.float32),
            torch.tensor(root_vel, dtype=torch.float32),
            torch.tensor(root_angular_vel, dtype=torch.float32),
            torch.tensor(root_pos_interpolated, dtype=torch.float32),
            torch.tensor(rotations_interpolated.as_quat(), dtype=torch.float32),
        )
