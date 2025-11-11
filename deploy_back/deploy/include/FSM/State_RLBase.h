// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"


// RL 控制状态类，负责与策略模型交互并控制机器人
class State_RLBase : public FSMState
{
public:
    // 构造函数，初始化 RL 状态，加载策略模型
    State_RLBase(int state_mode, std::string state_string);
    
    // 进入 RL 控制状态时调用
    // 主要功能：
    // 1. 设置关节增益（kp/kd）
    // 2. 更新机器人状态并重置环境
    // 3. 启动策略推理线程，周期性调用 env->step() 完成观测、推理、动作输出
    void enter()
    {
        // 设置每个关节的增益参数
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update(); // 同步底层状态
        env->reset();         // 重置 RL 环境

        // 启动策略推理线程
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // 初始化定时器
            const auto start = clock::now();
            auto sleepTill = start + dt;

            while (policy_thread_running)
            {
                env->step(); // 1. 收集观测 2. 策略推理 3. 输出动作

                // 控制推理周期
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    // 主循环调用，负责将策略输出的动作写入底层命令
    void run();
    
    // 退出 RL 控制状态时调用，安全关闭推理线程
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    // 策略模型路径解析工具，自动定位最新导出的 policy.onnx
    std::filesystem::path parser_policy_dir(std::filesystem::path policy_dir)
    {
        // 加载策略模型路径
        if (policy_dir.is_relative()) {
            policy_dir = param::proj_dir / policy_dir;
        }

        // 若无 exported 文件夹，则自动查找最新导出文件夹
        if (!std::filesystem::exists(policy_dir / "exported")) {
            auto dirs = std::filesystem::directory_iterator(policy_dir);
            std::vector<std::filesystem::path> dir_list;
            for (const auto& entry : dirs) {
                if (entry.is_directory()) {
                    dir_list.push_back(entry.path());
                }
            }
            if (!dir_list.empty()) {
                std::sort(dir_list.begin(), dir_list.end());
                // 从最新文件夹开始查找 exported
                for (auto it = dir_list.rbegin(); it != dir_list.rend(); ++it) {
                    if (std::filesystem::exists(*it / "exported")) {
                        policy_dir = *it;
                        break;
                    }
                }
            }
        }
        spdlog::info("Policy directory: {}", policy_dir.string());
        return policy_dir;
    }

    // RL 环境对象，负责观测、推理、动作管理
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    // 策略推理线程
    std::thread policy_thread;
    // 推理线程运行标志
    bool policy_thread_running = false;
};