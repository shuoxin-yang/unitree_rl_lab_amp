from __future__ import annotations
from dataclasses import MISSING
from isaaclab.utils import configclass


@configclass
class AMPCfg:

    #########################
    # Motion configurations #
    #########################

    # motion文件夹路径
    amp_data_path: str = MISSING

    # motion名称表，若需包含全部，则为 "*"
    amp_data_names: list[str] = ["*"]

    # motion权重，motion_k的权重表达式为weights[k]/sum_of_weights
    amp_data_weights: list[int] = [1]

    # motion噪声分布，若为1,则noise服从N(0,1),否则noise = scale * X~N(0,1)
    amp_data_noise_scale: float = 0.01

    # 机器人默认关节位置，若前期已处理，则置零，否则expert_motion = motion - default_joint_pos
    default_joint_pos: list[float] = MISSING

    ################################
    # Discriminator configurations #
    ################################

    # 隐藏层维度，神经网络为num_obs * 2 -> hidden_dims[i] -> 1
    hidden_dims: list[float] = [256, 128]

    # 启用随机失活
    use_dropout: bool = MISSING

    # 随机失活率，对应每个隐藏层的dropout值
    dropout_rate: list[float] = [0.0, 0.0]

    # 判别器更新频率，表示每 update_period 个仿真iterations更新一次判别器
    update_period: int = 100
