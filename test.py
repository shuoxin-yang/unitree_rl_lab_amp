from rsl_rl.utils.motion_loader import Motion
from rsl_rl.modules.discriminator import Discriminator
import numpy as np
import torch
import torch.nn as nn

motion_loader = Motion()
motion_loader.load_motions(
    motion_folder="./AMP_Motion",
    motion_files=["B15WA"],
    weights=[1.0],
    target_fps=50,
)
print(motion_loader.action_pairs[0])

discriminator = Discriminator(
    input_dim=motion_loader.action_pairs[0].shape[0],
    hidden_dims=[1024, 512],
    dropout_rate=0.01,
    device="cuda:0",
)

expert_reward_count = 0
policy_reward_count = 0
expert_reward = 0
policy_reward = 0
count = 0
optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

while True:
    discriminator.eval_mode()
    with torch.no_grad():
        expert_reward_count += expert_reward
        policy_reward_count += policy_reward
        expert_data = motion_loader.random_get_action_pair_batch(24)
        expert_reward, expert_logit = discriminator.forward(expert_data)
        policy_data = torch.randn_like(expert_data, dtype=torch.float32)
        policy_data = torch.tensor(policy_data, dtype=torch.float32).to(
            discriminator.device
        )
        policy_reward, policy_logit = discriminator.forward(policy_data)
    discriminator.train_mode()
    loss = discriminator.compute_loss(expert_data, policy_data, w_gp=10.0)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(discriminator.linear.parameters(), 1.0)
    optimizer.step()
    if count % 100 == 0 and count != 0:
        log_str = (
            "=========================================\n"
            f"Iteration: {count}\n"
            f"Current Expert Reward: {expert_reward.mean().item():.4f}, Current Policy Reward: {policy_reward.mean().item():.4f}\n"
            f"Current Expert Logit: {expert_logit.mean().item():.4f}, Current Policy Logit: {policy_logit.mean().item():.4f}\n"
            f"Mean Expert Reward: {expert_reward_count.mean().item()/100:.4f}, Mean Policy Reward: {policy_reward_count.mean().item()/100:.4f}\n"
            f"Loss: {loss.item():.4f}"
        )
        # 用 end='\r' 让光标回到第一行开头，下次打印覆盖当前内容
        print(log_str)
        expert_reward_count = 0
        policy_reward_count = 0
    print(count, end='\r')
    count += 1
