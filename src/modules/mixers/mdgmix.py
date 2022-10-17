import torch as th
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class G2ANet(nn.Module):
    def __init__(self, obs_dim, args):
        super(G2ANet, self).__init__()
        self.obs_dim = obs_dim
        self.n_agents = args.n_agents
        self.args = args
        
        self.encoding = nn.Linear(self.obs_dim, args.gru_hidden_dim)
        self.hard_bi_GRU = nn.GRU(args.gru_hidden_dim * 2, args.gru_hidden_dim, bidirectional=True)
        self.hard_encoding = nn.Linear(args.gru_hidden_dim * 2, 2)  # 乘2因为是双向GRU，hidden_state维度为2 * hidden_dim
    
    def forward(self, obs):
        batch_size, T = obs.shape[0], obs.shape[1]
        obs = obs.reshape(-1, self.obs_dim)
        size = obs.shape[0]
        obs_encoding = f.relu(self.encoding(obs))
        
        # Hard Attention
        h = obs_encoding.reshape(-1, self.n_agents, self.args.gru_hidden_dim)
        input_hard = th.Tensor().to(h.device)
        for i in range(self.n_agents):
            h_i = th.cat([h[:, i].unsqueeze(1).repeat(1, self.n_agents, 1), h], dim=-1)
            input_hard = th.cat([input_hard, h_i], dim=0)
        input_hard = input_hard.view(self.n_agents, batch_size, T, self.n_agents, self.args.gru_hidden_dim * 2).permute(3, 1, 2, 0, 4)
        h_t = th.zeros((2 * 1, batch_size * self.n_agents, self.args.gru_hidden_dim)).to(h.device)
        
        h_hard = []
        for t in range(T):
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
        hard_weights = hard_weights[:, 1].view(batch_size, T, self.n_agents, self.n_agents)
        
        return obs_encoding, hard_weights
        
class Attention_aggregator(nn.Module):
    def __init__(self, feature_dim, embed_dim, concat):
        super(Attention_aggregator, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.concat = concat
        self.encoder_layer = nn.Linear(self.feature_dim * 2 if self.concat or 
            isinstance(self.aggregator, LSTM_aggregator) else self.feature_dim, self.embed_dim, bias=False)
            
    def forward(self, node_features, nodes, adj_list):
        attention = th.matmul(node_features, node_features.permute([0, 2, 1]))
        attention[adj_list == 0] = -9999999
        masked_attention = f.softmax(attention, dim=-1)
        combined_features = th.matmul(masked_attention, node_features)
        if self.concat:
            combined_features = th.cat([node_features, combined_features], dim=-1)
        output = f.relu(self.encoder_layer(combined_features))
        return output

class MDGMixer(nn.Module):
    def __init__(self, args):
        super(MDGMixer, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.head_num = args.head_num
        
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = int(np.prod(args.observation_shape))
        self.feature_dim = args.gru_hidden_dim
        self.hard_attention = G2ANet(self.obs_dim, args)
        self.encoder = nn.ModuleList([Attention_aggregator(self.feature_dim, self.hidden_dim, args.concat) for _ in range(self.head_num)])
        self.output_layer = nn.Linear(self.hidden_dim, 1)

        self.hyper_weight_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.head_num * 1)
        )
        self.hyper_const_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, 1)
        )
        
    def forward(self, agent_qs, states, obs):
        bs = agent_qs.size(0)
        sl = agent_qs.size(1)
        obs_encoding, hard_weights = self.hard_attention(obs)
        agent_qs = agent_qs.reshape(-1, agent_qs.size(-1))
        states = states.reshape(-1, states.size(-1))
        obs_encoding = obs_encoding.reshape(bs * sl, self.n_agents, -1)
        hard_weights = hard_weights.reshape(-1, self.n_agents, self.n_agents)
        nodes = th.LongTensor(list(range(self.n_agents)))
        enc_outputs = []
        for h in range(self.head_num):
            enc_output = self.encoder[h](obs_encoding, nodes, hard_weights)
            enc_outputs.append(enc_output)
        enc_outputs = th.stack(enc_outputs, dim=-2)
        output_weight = self.output_layer(enc_outputs).squeeze()
        output_weight = f.softmax(output_weight, dim=-2)
        qs_tot = th.bmm(agent_qs.unsqueeze(1), output_weight)

        hyper_weight = th.abs(self.hyper_weight_layer(states).view(-1, self.head_num, 1))
        hyper_const = self.hyper_const_layer(states).view(-1, 1, 1)
        q_tot = th.bmm(qs_tot, hyper_weight) + hyper_const
        return q_tot.view(bs, sl, 1)