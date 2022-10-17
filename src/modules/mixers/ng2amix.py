import torch as th
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


# 输入所有agent的obs，输出所有agent的动作概率分布
class newG2AMixer(nn.Module):
    def __init__(self, scheme, args):
        super(newG2AMixer, self).__init__()
        self.n_agents = args.n_agents
        self.scheme = scheme
        input_shape = self._get_input_shape()
        # Encoding
        # self.encoding = nn.Linear(input_shape, args.gru_hidden_dim)  # 对所有agent的obs编码
        self.encoding = nn.Sequential(nn.Linear(input_shape, 2 * args.gru_hidden_dim), nn.ReLU(), nn.Linear(2 * args.gru_hidden_dim, args.gru_hidden_dim))

        # Hard
        # GRU输入[[h_i,h_1],[h_i,h_2],...[h_i,h_n]]与[0,...,0]，输出[[h_1],[h_2],...,[h_n]]与[h_n]， h_j表示了agent j与agent i的关系
        # 输入的inputs维度为(n_agents, batch_size * n_agents, gru_hidden_dim * 2)，
        # 即对于batch_size条数据，输入每个agent与其他n_agents个agents的hidden_state的连接
        self.hard_bi_GRU = nn.GRU(args.gru_hidden_dim * 2, args.gru_hidden_dim, bidirectional=True)
        # 对h_j进行分析，得到agent j对于agent i的权重，输出两维，经过gumble_softmax后取其中一维即可，如果是0则不考虑agent j，如果是1则考虑
        self.hard_encoding = nn.Linear(args.gru_hidden_dim * 2, 2)  # 乘2因为是双向GRU，hidden_state维度为2 * hidden_dim

        # Soft
        self.q = nn.Linear(args.gru_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.gru_hidden_dim, args.attention_dim, bias=False)
        # self.v = nn.Linear(args.gru_hidden_dim, args.attention_dim)

        # Hyper_net
        self.state_dim = int(np.prod(args.state_shape))
        self.head_num = args.head_num
        self.hyper_weight_layer = nn.Sequential(
            nn.Linear(self.state_dim, args.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hyper_hidden_dim, self.n_agents)
        )
        self.hyper_const_layer = nn.Sequential(
            nn.Linear(self.state_dim, args.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hyper_hidden_dim, 1)
        )        

        self.args = args
        self.input_shape = input_shape

    def forward(self, agent_qs, states, obs):
        # agent_qs: (batch_size, T, n_agents), obs: (batch_size, T, n_agents, obs_dim)
        batch_size, T = obs.shape[0], obs.shape[1]
        states = states.reshape(-1, self.state_dim)
        obs = obs.reshape(-1, self.scheme["obs"]["vshape"])
        agent_qs = agent_qs.reshape(-1, self.n_agents, 1)
        
        size = obs.shape[0]  # batch_size * T * n_agents
        # 先对obs编码
        obs_encoding = f.relu(self.encoding(obs))

        # Hard Attention，GRU和GRUCell不同，输入的维度是(序列长度,batch_size, dim)
        if self.args.hard:
            # Hard Attention前的准备
            h = obs_encoding.reshape(-1, self.args.n_agents, self.args.gru_hidden_dim)  # (batch_size * T, n_agents, gru_hidden_dim)
            '''
            input_hard = []
            for i in range(self.args.n_agents):
                h_i = h[:, i]  # (batch_size * T, 1, gru_hidden_dim)
                h_hard_i = []
                for j in range(self.args.n_agents):  # 对于agent i，把自己的h_i与其他agent的h分别拼接
                    # if j != i:
                    h_hard_i.append(th.cat([h_i, h[:, j]], dim=-1))
                # j 循环结束之后，h_hard_i是一个list里面装着n_agents个维度为(batch_size * T, 1, gru_hidden_dim * 2)的tensor
                h_hard_i = th.stack(h_hard_i, dim=0)
                input_hard.append(h_hard_i)
            # i循环结束之后，input_hard是一个list里面装着n_agents个维度为(n_agents, batch_size * T, 1, gru_hidden_dim * 2)的tensor
            input_hard = th.stack(input_hard, dim=-2)
            # 最终得到维度(n_agents, batch_size * T, n_agents, gru_hidden_dim * 2)，可以输入了
            input_hard = input_hard.view(self.n_agents, -1, self.args.gru_hidden_dim * 2)   # (n_agents, batch_size * T * n_agents, gru_hidden_dim * 2)
            '''
            input_hard = th.Tensor()
            if self.args.use_cuda:
                input_hard = input_hard.cuda()
            for i in range(self.n_agents):
                h_i = th.cat([h[:, i].unsqueeze(1).repeat(1, self.n_agents, 1), h], dim=-1)
                input_hard = th.cat([input_hard, h_i], dim=0)
            input_hard = input_hard.view(self.n_agents, batch_size, T, self.n_agents, self.args.gru_hidden_dim * 2).permute(3, 1, 2, 0, 4)
            h_t = th.zeros((2 * 1, batch_size * self.n_agents, self.args.gru_hidden_dim))  # 因为是双向GRU，每个GRU只有一层，所以第一维是2 * 1
            if self.args.use_cuda:
                h_t = h_t.cuda()
            h_hard = []
            for t in range(T):
                '''
                h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents, batch_size * T * n_agents, gru_hidden_dim * 2)
                h_hard = h_hard.permute(1, 0, 2)  # (batch_size * T * n_agents, n_agents, gru_hidden_dim * 2)
                h_hard = h_hard.reshape(-1, self.args.gru_hidden_dim * 2)  # (batch_size * T * n_agents * n_agents, gru_hidden_dim * 2)
                '''
                input_hard_t = input_hard[:, :, t].reshape(self.n_agents, -1, self.args.gru_hidden_dim * 2)
                h_hard_t, h_t = self.hard_bi_GRU(input_hard_t, h_t)
                h_hard.append(h_hard_t)
            h_hard = th.stack(h_hard, dim=-2)
            h_hard = h_hard.reshape(self.n_agents, batch_size, self.n_agents, T, self.args.gru_hidden_dim * 2)
            h_hard = h_hard.permute(1, 3, 2, 0, 4)
            h_hard = h_hard.reshape(-1, self.args.gru_hidden_dim * 2)

            # 得到hard权重
            hard_weights = self.hard_encoding(h_hard)
            hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
            # hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
            hard_weights = hard_weights[:, 1].view(-1, self.n_agents, self.n_agents)

        else:
            hard_weights = th.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents))
            if self.args.use_cuda:
                hard_weights = hard_weights.cuda()
            hard_weights = hard_weights.view(-1, self.n_agents, self.n_agents)

        # Soft Attention
        q = self.q(obs_encoding).reshape(-1, self.n_agents, self.args.attention_dim)  # (batch_size * T, n_agents, args.attention_dim)
        k = self.k(obs_encoding).reshape(-1, self.n_agents, self.args.attention_dim)  # (batch_size * T, n_agents, args.attention_dim)
        # v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        # v = agent_qs
        k = k.permute(0, 2, 1)
        agent_qs = agent_qs.permute(0, 2, 1)
        # for i in range(self.args.n_agents):
            # q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # agent i的q，(batch_size * T, 1, args.attention_dim)
            # k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]  # 对于agent i来说，其他agent的k
            # v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]  # 对于agent i来说，其他agent的v

            # k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, args.attention_dim)
            # k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, args.attention_dim， n_agents - 1)
            # v_i = torch.stack(v_i, dim=0)
            # v_i = v_i.permute(1, 2, 0)
            
        # (batch_size * T, n_agents, attention_dim) * (batch_size * T, attention_dim，n_agents) = (batch_size * T, n_agents，n_agents)
        score = th.bmm(q, k)
        # 归一化
        scaled_score = score / np.sqrt(self.args.attention_dim)
        # scaled_score[hard_weights==0] = float("-inf")
        # softmax得到权重
        soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size * T，n_agents, n_agents)
        # 加权求和，注意三个矩阵的最后一维是n_agents - 1维度，得到(batch_size * T, n_agents)
        x = (agent_qs * soft_weight * hard_weights).sum(dim=-1)
            # x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)

        # 合并每个agent的h与x
        # x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # (batch_size * n_agents, args.attention_dim)
        # final_input = torch.cat([h_out, x], dim=-1)
        # output = self.decoding(final_input)

        # sum
        # q_tot = th.sum(x, dim=1)

        # hyper_net
        x = f.relu(x).view(-1, 1, self.n_agents)
        hyper_weight = th.abs(self.hyper_weight_layer(states).view(-1, self.n_agents, 1))
        hyper_const = self.hyper_const_layer(states).view(-1, 1, 1)
        q_tot = th.bmm(x, hyper_weight) + hyper_const
        return q_tot.view(batch_size, T, 1)
    
    def _get_input_shape(self):
        return self.scheme["obs"]["vshape"]
