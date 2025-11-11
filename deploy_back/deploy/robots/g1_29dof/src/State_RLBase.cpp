#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    spdlog::info("Initializing State_{}...", state_string);
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            (int)FSMMode::Passive
        )
    );
}

void State_RLBase::run()
{
    // 1. 获取策略推理后的动作（关节目标位置）
    auto action = env->action_manager->processed_actions();
    // 2. 遍历所有关节，将动作写入底层命令
    //std::cout<<"init"<<std::endl;
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        // motor_cmd 是底层电机命令数组，q() 表示目标关节位置
        // joint_ids_map[i] 是第 i 个关节的底层索引
        // action[i] 是策略输出的第 i 个关节目标位置
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
        // std::cout<<"kp:"<<lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kp()<<std::endl;
        // std::cout<<"kd:"<<lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].kd()<<std::endl;
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].mode(1);
        // std::cout<<"mode"<<lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].mode()<<std::endl;
        // std::cout<<"mode_pr"<<lowcmd->msg_.mode_pr()<<std::endl;
    }
    // 3. 机器人底层会周期性读取 motor_cmd，实现动作控制
}