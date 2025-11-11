// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>

namespace isaaclab
{

class ObservationManager;
class ActionManager;


// RL环境主类，负责管理观测、动作、机器人底层和策略推理
class ManagerBasedRLEnv
{
public:
    // 构造函数
    // 1. 解析配置文件，初始化步长、关节映射、默认位置、刚度/阻尼等参数
    // 2. 创建底层机器人对象
    // 3. 初始化动作管理器和观测管理器
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // 解析仿真步长
        this->step_dt = cfg["step_dt"].as<float>();
        // 关节映射与状态初始化
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // 默认关节位置
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // 关节刚度与阻尼
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update(); // 同步底层状态

        // 初始化动作与观测管理器
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    // 环境重置，清空计数器并重置动作/观测管理器
    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        action_manager->reset();
        observation_manager->reset();
    }

    // 环境步进函数
    // 1. 更新机器人底层状态
    // 2. 计算观测
    // 3. 策略推理（alg->act）获取动作
    // 4. 处理动作（写入底层命令）
    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute(); // 获取观测
        
        auto action = alg->act(obs);               // 策略推理
        action_manager->process_action(action);    // 输出动作
    }

    float step_dt; // 环境步长（控制周期）
    
    YAML::Node cfg; // 环境配置

    std::unique_ptr<ObservationManager> observation_manager; // 观测管理器
    std::unique_ptr<ActionManager> action_manager;           // 动作管理器
    std::shared_ptr<Articulation> robot;                     // 机器人底层对象
    std::unique_ptr<Algorithms> alg;                         // 策略推理器（如 ONNX 推理）
    long episode_length = 0;                                 // 当前回合步数
    float global_phase = 0.0f;                               // 全局相位（可用于步态等）
};

};