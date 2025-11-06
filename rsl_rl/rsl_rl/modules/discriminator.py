import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        device: str = "cuda:0",
    ):
        """
        Discriminator network for RSL-RL.
        input_dim: Dimension of the input features. For g1 is state-state pairs, depends on the obs.
        hidden_dim: List of hidden layer dimensions.
        reward_scale: Scaling factor for the output reward.
        """
        super(Discriminator, self).__init__()
        # 初始化参数
        self.device = device
        # # 添加数据归一化层
        # self.running_mean = torch.zeros(input_dim, device=device)
        # self.running_var = torch.ones(input_dim, device=device)
        # self.batch_norm = nn.BatchNorm1d(input_dim).to(device)
        # self.batch_norm.eval()  # 固定归一化层参数
        # 构建网络层
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        # 组合隐藏层并移动到指定设备
        self.trunk = nn.Sequential(*layers).to(device)
        # 链接最后的隐藏层与输出层，输出为标量
        self.linear = nn.Linear(current_dim, 1).to(device)
        # 启动训练模式
        self.trunk.train()
        self.linear.train()

    def forward(
        self, state_trans_pair: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，获取判别器输出
        参数：
            state_trans_pair: 形状为 (batch_size, input_dim) 的张量，表示每一个环境的状态-状态转移对

        返回：
            reward: 形状为 (batch_size) 的张量，表示每一个环境的判别器奖励
        """
        # 确保输入是2D张量
        if state_trans_pair.dim() == 1:
            state_trans_pair = state_trans_pair.unsqueeze(0)

        # 通过网络获取logit
        h = self.trunk(state_trans_pair)
        logit = self.linear(h).squeeze(-1)
        # 计算奖励
        reward = torch.max(torch.tensor(0), 1 - 0.25 * ((logit - 1) ** 2))
        return reward, logit

    def compute_loss(
        self,
        expert_state_trans_pair: torch.Tensor,
        policy_state_trans_pair: torch.Tensor,
        w_gp: float = 10.0,
    ) -> torch.Tensor:
        """
        计算判别器的损失函数
        参数：
            expert_state_trans_pair: 形状为 (batch_size, input_dim) 的张量，表示专家数据的状态-状态转移对
            policy_state_trans_pair: 形状为 (batch_size, input_dim) 的张量，表示策略数据的状态-状态转移对
            w_gp: 梯度惩罚的权重, 默认为10.0
        返回：
            loss: 标量张量，表示判别器的损失
        """
        # 获取专家数据和策略数据的logit, 分别计算损失
        _, d_policy = self.forward(policy_state_trans_pair)
        # loss_policy = self.loss_function(d_policy, torch.ones_like(d_policy))
        loss_policy = torch.mean((d_policy + 1) ** 2)
        _, d_expert = self.forward(expert_state_trans_pair)
        # loss_expert = self.loss_function(d_expert, torch.full_like(d_expert, -1))
        loss_expert = torch.mean((d_expert - 1) ** 2)
        # 计算梯度损失
        loss_gp = self.compute_gradient_penalty(
            expert_state_trans_pair, policy_state_trans_pair, w_gp
        )
        # 汇总损失
        loss_sum = loss_expert + loss_policy + 0.5 * loss_gp
        return loss_sum

    def train_mode(self):
        """
        切换到训练模式
        """
        self.trunk.train()
        self.linear.train()

    def eval_mode(self):
        """
        切换到评估模式
        """
        self.trunk.eval()
        self.linear.eval()

    def compute_gradient_penalty(
        self, expert_pairs: torch.Tensor, policy_pairs: torch.Tensor, w_gp
    ) -> torch.Tensor:
        """
        计算梯度惩罚项
        参数：
            expert_pairs: 形状为 (batch_size, input_dim) 的张量，表示专家数据的状态-状态转移对
            policy_pairs: 形状为 (batch_size, input_dim) 的张量，表示策略数据的状态-状态转移对
            w_gp: 梯度惩罚的权重
        返回：
            gradient_penalty: 标量张量，表示梯度惩罚项
        """
        # 计算总环境数
        batch_size = expert_pairs.size(0)
        # 生成差值随机参数
        alpha = torch.rand(batch_size, 1).to(expert_pairs.device)
        # 计算插值数据
        mix_pair = alpha * expert_pairs + (1 - alpha) * policy_pairs
        # 设置插值数据需要梯度
        mix_pair.requires_grad_(True)
        # 计算插值数据的判别器输出
        _, d_mix_pair = self.forward(mix_pair)
        # 计算梯度
        gradients = autograd.grad(
            outputs=d_mix_pair,
            inputs=mix_pair,
            grad_outputs=torch.ones(d_mix_pair.size()).to(expert_pairs.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = w_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty
