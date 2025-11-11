# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.utils import AMPCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    isAMP = True
    amp = AMPCfg(
        amp_data_path="/home/yangyuhui/unitree_rl_lab_amp/AMP_Motion",
        amp_data_names=["Female_Walk"],
        # amp_data_names = ["AMPdebug"],
        amp_data_weights=[1.0],
        amp_data_noise_scale=0.01,
        default_joint_pos=[
            -0.1000,
            -0.1000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.3000,
            0.3000,
            0.3000,
            0.3000,
            -0.2000,
            -0.2000,
            0.2500,
            -0.2500,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.9700,
            0.9700,
            0.1500,
            -0.1500,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        # default_joint_pos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        hidden_dims=[1024, 512],
        use_dropout=False,
        dropout_rate=[0.2, 0.2],
        update_period=2,
    )
