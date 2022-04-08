#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions) # 网络直接输出Logits而并非Softmax结果 \
                                            # 然后使用CrossEntropyLoss()计算，注意与NLLoss()区分.
        )

    def forward(self, x):
        return self.net(x)

# 用于保存书中Figure 2中的每一个小方格.
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
# 用于将Figure 2中每一个小方格拼接成一个Episode， 在本代码实现中discount gamma = 1.
Episode = namedtuple('Episode', field_names=['reward', 'steps'])


def iterate_batches(env, net, batch_size):
    """生成器，不断产生batch_size大小的Episodes集合
    :param env: gym环境
    :param net: 网络
    :param batch_size:
    :return: 通过yield产生了函数生成器，将本函数放到for循环中 \
            将不断产生新的Batch Size大小的Episode List集合.
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1) 
    while True:
        obs_v = torch.FloatTensor([obs])  # 使用'[]'目的是增加一个Batch维度，否则Torch报错
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action) # 记录一次环境+选择的动作
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps) # 游戏结束，将当前episode保存.
            batch.append(e)
            # 重新初始化必要参数，因为生成器等会儿会回到while循环中，到不了while上方代码.
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    """CrossEntropy方法的核心筛选出elites
    :param batch: 初始的Batch Size个episodes
    :param percentile: 筛选的百分数
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        # train_obs将不再有原来的Batch维度! 而是全新的Batch维度.
        # N: 符合要求的所有单次观测，一般远大于BATCH_SIZE(16)
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v) # CrossEntropyLoss接收Logits
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199: # gym-PortCart分数上限为200
            print("Solved!")
            break
    writer.close()
