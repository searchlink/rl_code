#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 0021 11:31
# @Author  : skydm
# @email   : 
# @File    : mrp.py
# @Software: PyCharm
# @desc    : 马尔科夫奖励过程

import numpy as np

num_states = 7
i_to_n = {}
i_to_n[0] = 'C1'
i_to_n[1] = 'C2'
i_to_n[2] = 'C3'
i_to_n[3] = 'Pass'
i_to_n[4] = 'Pub'
i_to_n[5] = 'FB'
i_to_n[6] = 'Sleep'

n_to_i = {}
for i, name in i_to_n.items():
    n_to_i[name] = i

# 转移概率
#   C1   C2   C3   Pass  Pub  FB  Sleep
Pss = [
   [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0 ],
   [ 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2 ],
   [ 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0 ],
   [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
   [ 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0 ],
   [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
   [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
]

Pss = np.array(Pss)

rewards = [-2, -2, -2, 10, 1, -1, 0]
gamma = 0.5

# 采样得到如下状态序列：
chains =[
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB",\
     "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

# 辅助函数
# 根据马尔科夫决策过程的数据来设置奖励和转移概率
def generate_state_action_state_prob_dict(P, s, a, s_quote, prob=1.0):
    '''马尔科夫决策过程'''
    sas = '_'.join([s, a, s_quote])
    P[sas] = prob

def generate_state_action_reward_dict(R, s, a, r=1.0):
    '''马尔科夫决策过程'''
    sa = '_'.join([s, a])
    R[sa] = r

def generate_pi_dict(Pi, s, a, prob=0.5):
    sa = '_'.join([s, '选择', a, '概率'])
    Pi[sa] = prob

# 计算回报(return): 从某一状态S_t开始采样知道终止状态时所有奖励的有衰减之和。
def compute_return(start_index, chain, gamma=0.5):
    '''
    计算一个马尔科夫奖励过程中某状态的收获值
    Args:
        start_index 要计算的状态在状态序列中的位置
        chain 要计算的状态序列
        gamma 衰减系数
    Returns：
        retrn 某一状态的累积收获值
    '''
    print(chain)
    R = 0.
    for index, char in enumerate(range(start_index, len(chain))):
        R += np.power(gamma, index) * rewards[n_to_i[chain[char]]]

    return R

print(compute_return(0, chains[3], gamma = 0.5))
print(compute_return(3,chains[3]))

# 状态价值函数的计算(矩阵运算)
def compute_state_value(Pss, rewards, gamma=0.5):
    ''' v = R + gamma * P * v
    :param Pss: 状态转移概率矩阵
    :param rewards: 即时奖励list
    :param gamma: 衰减因子
    :return: 各状态价值(期望值， 收敛状态)
    '''
    reward = np.array(rewards).reshape((-1, 1))
    values = np.dot(np.linalg.inv(np.eye(7, 7) - gamma * Pss), rewards)
    return values

values = compute_state_value(Pss, rewards, gamma=0.99999)
print(values)


