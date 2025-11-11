#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

// 机器人底层命令指针
std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
// 机器人底层状态指针
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
// 键盘输入指针
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

// 初始化 FSM 状态，包括底层命令和状态，连接机器人
void init_fsm_state()
{
    // 检查 lowcmd 通道是否被其他进程占用
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6); // 等待 200ms，保证 DDS 通信初始化
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown(); // 关闭 go2 机器人进程
        // exit(0); // 可选：直接退出
    }
    // 创建底层命令和状态对象
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection(); // 阻塞直到连接成功
    spdlog::info("Connected to robot.");
    // Print a brief summary of the first received LowState
    
}


int main(int argc, char** argv)
{
    // 1. 加载参数
    auto vm = param::helper(argc, argv); // 解析命令行参数

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // 2. 初始化 Unitree DDS 通信
    // 打印并记录 DDS 初始化参数（domain, network/interface）以便调试
    std::string network_if = vm["network"].as<std::string>();
    // network_if = "enp131s0";
    int domain_id = 0;
    if(network_if == "lo"){
        domain_id = 1;
    }

    spdlog::info("Initializing ChannelFactory: domain={} network='{}'", domain_id, network_if);
    unitree::robot::ChannelFactory::Instance()->Init(domain_id, network_if);
    spdlog::info("ChannelFactory::Init called");

    // 3. 初始化 FSM 状态（连接机器人）
    init_fsm_state();

    // 4. 设置机器人模式为 29dof
    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof 模式编号
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type."); // 机器人类型不匹配
        exit(-1);
    }

    // 5. 初始化有限状态机 FSM
    auto & joy = FSMState::lowstate->joystick; // 获取手柄对象
    auto fsm = std::make_unique<CtrlFSM>(new State_Passive(FSMMode::Passive)); // 初始为被动模式

    // 注册状态切换：L2+Up 切换到 FixStand
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return joy.LT.pressed && joy.up.pressed; }, // L2 + Up
            (int)FSMMode::FixStand
        )
    );
    fsm->add(new State_FixStand(FSMMode::FixStand)); // 添加 FixStand 状态

    // 注册状态切换：R1+X 切换到 RL 控制模式
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return joy.RB.pressed && joy.X.pressed; }, // R1 + X
            FSMMode::Velocity
        )
    );
    fsm->add(new State_RLBase(FSMMode::Velocity, "Velocity")); // 添加 RL 控制状态
    
    // 注册状态切换：R1+Y 切换到 被动模式
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return joy.RB.pressed && joy.Y.pressed; }, // R1 + Y
            FSMMode::Passive
        )
    );
    // fsm->add(new State_Passive(FSMMode::Passive)); // 添加 ps 控制状态

    // 6. 用户提示
    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    // 7. 主循环（保持进程存活）
    while (true)
    {
        if(joy.RB.pressed && joy.Y.pressed){
            
        }
        //printJoystickInf(joy);
        sleep(1); // 每秒休眠，实际控制逻辑在 FSM 内部
    }

    return 0;
}

