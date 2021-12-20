import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from actor_critic import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PADDPG(object):
    def __init__(self, state_dim, action_dim, max_action, min_action, td3, exploration="EG", discount=0.99, tau=1e-4):
        self.exploration = exploration
        self.td3 = td3

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.ext_critic = Critic(state_dim, action_dim, td3).to(device)
        self.ext_critic_target = copy.deepcopy(self.ext_critic)
        self.ext_critic_optimizer = torch.optim.Adam(self.ext_critic.parameters(), lr=1e-3)

        if not self.exploration == "EG":
            self.int_critic = Critic(state_dim, action_dim, td3).to(device)
            self.int_critic_target = copy.deepcopy(self.int_critic)
            self.int_critic_optimizer = torch.optim.Adam(self.int_critic.parameters(), lr=1e-3)


        self.discount = discount
        self.tau = tau

        self.max_p = torch.FloatTensor(max_action).to(device)
        self.min_p = torch.FloatTensor(min_action).to(device)
        self.rng = (self.max_p - self.min_p).detach()

        # td3 hyper prameters
        if self.td3:
            self.policy_noise = 0.2
            self.noise_clip=0.5
            self.policy_freq=2
            self.total_it = 0


    def invert_gradient(self, delta_a, current_a):
        index = delta_a > 0
        delta_a[index] *= (index.float() * (self.max_p - current_a) / self.rng)[index]
        delta_a[~index] *= ((~index).float() * (current_a - self.min_p) / self.rng)[~index]
        return delta_a

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        p = self.actor(state)
        np_max = self.max_p.cpu().data.numpy()
        np_min = self.min_p.cpu().data.numpy()
        return np.clip(p.cpu().data.numpy().flatten(), np_min, np_max)

    # TODO: 内部報酬を利用した場合の学習安定化
    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, int_reward, n_step, ex_n_step, not_done = replay_buffer.sample(batch_size)

        if self.td3:
            self.total_it += 1
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(self.min_p, self.max_p)

            ext_target_q1, ext_target_q2 = self.ext_critic_target(next_state, next_action)
            ext_target_Q = torch.min(ext_target_q1, ext_target_q2)
            #ext_target_Q = self.ext_critic_target(next_state, self.actor_target(next_state))

            # 探索法別にtargetQの計算を分ける
            if self.exploration == "EG":
                ext_target_Q = reward + ((1 - not_done) * self.discount * ext_target_Q).detach()
                beta = 0.2

                ext_mixed_Q = beta * n_step + (1 - beta) * ext_target_Q
                # 内部報酬のn_stepはエピソード終了までの軌跡の内部報酬が無意味になってしまう
                # mixed_q = beta * (n_step + ex_n_step) + (1 - beta) * target_Q
                ext_current_q1, ext_current_q2 = self.ext_critic(state, action)
                ext_critic_loss = F.mse_loss(ext_current_q1, ext_mixed_Q) + F.mse_loss(ext_current_q2, ext_mixed_Q)

                self.ext_critic_optimizer.zero_grad()
                ext_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ext_critic.parameters(), 10)
                self.ext_critic_optimizer.step()

            else:
                int_target_q1, int_target_q2 = self.int_critic_target(next_state, self.actor_target(next_state))
                int_target_Q = torch.min(int_target_q1, int_target_q2)

                ext_target_Q = reward + ((1-not_done) * self.discount * ext_target_Q).detach()
                int_target_Q = int_reward + ((1-not_done) * self.discount * int_target_Q).detach()

                ext_current_q1, ext_current_q2 = self.ext_critic(state, action)
                int_current_q1, int_current_q2 = self.int_critic(state,action)
                beta = 0.2

                ext_mixed_Q = beta * n_step + (1 - beta) * ext_target_Q
                # 内部報酬のn_stepはエピソード終了までの軌跡の内部報酬が無意味になってしまう
                # mixed_q = beta * (n_step + ex_n_step) + (1 - beta) * target_Q

                ext_critic_loss = F.mse_loss(ext_current_q1, ext_mixed_Q) + F.mse_loss(ext_current_q2, ext_mixed_Q)
                int_critic_loss = F.mse_loss(int_current_q1, int_target_Q) + F.mse_loss(int_current_q2, int_target_Q)

                self.ext_critic_optimizer.zero_grad()
                ext_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ext_critic.parameters(), 10)
                self.ext_critic_optimizer.step()

                self.int_critic_optimizer.zero_grad()
                int_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.int_critic.parameters(), 10)
                self.int_critic_optimizer.step()

            if self.total_it % self.policy_freq == 0:
                current_a = Variable(self.actor(state))
                current_a.requires_grad = True

                # L=(intQ(s,a)+extQ(s,a))の平均をactor_lossとした場合
                if self.exploration == "EG":
                    actor_loss = self.ext_critic.Q1(state,current_a)
                else:
                    actor_loss = self.ext_critic.Q1(state,current_a)+self.int_critic.Q1(state,current_a)
                actor_loss = actor_loss.mean()

                self.ext_critic.zero_grad()
                if not self.exploration == "EG":
                    self.int_critic.zero_grad()
                # TODO: current_aの勾配が正しいかの確認
                actor_loss.backward()
                delta_a = copy.deepcopy(current_a.grad.data)
                delta_a = self.invert_gradient(delta_a, current_a)
                current_a = self.actor(Variable(state))
                out = -torch.mul(delta_a, current_a)
                self.actor.zero_grad()
                out.backward(torch.ones(out.shape).to(device))
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            ext_target_Q = self.ext_critic_target.Q1(next_state, self.actor_target(next_state))

            # 探索法別にtargetQの計算を分ける
            if self.exploration == "EG":
                ext_target_Q = reward + ((1 - not_done) * self.discount * ext_target_Q).detach()

                ext_current_Q = self.ext_critic.Q1(state, action)
                beta = 0.2

                ext_mixed_Q = beta * n_step + (1 - beta) * ext_target_Q
                # 内部報酬のn_stepはエピソード終了までの軌跡の内部報酬が無意味になってしまう
                # mixed_q = beta * (n_step + ex_n_step) + (1 - beta) * target_Q

                ext_critic_loss = F.mse_loss(ext_current_Q, ext_mixed_Q)

                self.ext_critic_optimizer.zero_grad()
                ext_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ext_critic.parameters(), 10)
                self.ext_critic_optimizer.step()

            else:
                int_target_Q = self.int_critic_target.Q1(next_state, self.actor_target(next_state))

                ext_target_Q = reward + ((1 - not_done) * self.discount * ext_target_Q).detach()
                int_target_Q = int_reward + ((1 - not_done) * self.discount * int_target_Q).detach()

                ext_current_Q = self.ext_critic.Q1(state, action)
                int_current_Q = self.int_critic.Q1(state, action)
                beta = 0.2

                ext_mixed_Q = beta * n_step + (1 - beta) * ext_target_Q
                # 内部報酬のn_stepはエピソード終了までの軌跡の内部報酬が無意味になってしまう
                # mixed_q = beta * (n_step + ex_n_step) + (1 - beta) * target_Q

                ext_critic_loss = F.mse_loss(ext_current_Q, ext_mixed_Q)
                int_critic_loss = F.mse_loss(int_current_Q, int_target_Q)

                self.ext_critic_optimizer.zero_grad()
                ext_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ext_critic.parameters(), 10)
                self.ext_critic_optimizer.step()

                self.int_critic_optimizer.zero_grad()
                int_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.int_critic.parameters(), 10)
                self.int_critic_optimizer.step()

            current_a = Variable(self.actor(state))
            current_a.requires_grad = True

            # L=(intQ(s,a)+extQ(s,a))の平均をactor_lossとした場合
            if self.exploration == "EG":
                actor_loss = self.ext_critic.Q1(state,current_a)
            else:
                actor_loss = self.ext_critic.Q1(state,current_a)+self.int_critic.Q1(state,current_a)
            actor_loss = actor_loss.mean()

            self.ext_critic.zero_grad()
            if not self.exploration == "EG":
                self.int_critic.zero_grad()
            # TODO: current_aの勾配が正しいかの確認
            actor_loss.backward()
            delta_a = copy.deepcopy(current_a.grad.data)
            delta_a = self.invert_gradient(delta_a, current_a)
            current_a = self.actor(Variable(state))
            out = -torch.mul(delta_a, current_a)
            self.actor.zero_grad()
            out.backward(torch.ones(out.shape).to(device))
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        for param, target_param in zip(self.ext_critic.parameters(), self.ext_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if not self.exploration == "EG":
            for param, target_param in zip(self.int_critic.parameters(), self.int_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # itemでpythonのint型として返す
        if self.exploration == "EG":
            total_loss = ext_critic_loss
        else:
            total_loss = ext_critic_loss + int_critic_loss

        return ext_mixed_Q.mean().item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
