#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 0021 13:47
# @Author  : skydm
# @email   : 
# @File    : mdp.py
# @Software: PyCharm
# @desc    : 马尔科夫决策过程
#
# ：贝尔曼方程

import numpy as np
from utils import generate_state_action_state_prob_dict, generate_state_action_reward_dict, generate_pi_dict

# 定义状态
S = ['浏览手机中', '第一节课', '第二节课', '第三节课', '休息中']
# 定义行为
A = ['浏览手机','学习','离开浏览','泡吧','退出学习']
# 定义基于行为的状态转移概率矩阵Pss'a
P = {}
# 定义基于状态和行为的奖励函数Rsa
R = {}
# 衰减因子gamma
gamma = 1.0
# 策略
Pi = {}

# 基于行为的状态转移概率矩阵Pss'a
generate_state_action_state_prob_dict(P, S[0], A[0], S[0])  # '浏览手机中_离开浏览_第一节课': 1.0
generate_state_action_state_prob_dict(P, S[0], A[2], S[1])  # '浏览手机中_浏览手机_浏览手机中': 1.0
generate_state_action_state_prob_dict(P, S[1], A[0], S[0])  # '第一节课_浏览手机_浏览手机中': 1.0
generate_state_action_state_prob_dict(P, S[1], A[1], S[2])  # '第一节课_学习_第二节课': 1.0
generate_state_action_state_prob_dict(P, S[2], A[1], S[3])  # '第二节课_学习_第三节课': 1.0
generate_state_action_state_prob_dict(P, S[2], A[4], S[4])  # '第二节课_退出学习_休息中': 1.0
generate_state_action_state_prob_dict(P, S[3], A[1], S[4])  # '第三节课_学习_休息中': 1.0
generate_state_action_state_prob_dict(P, S[3], A[3], S[1], prob=0.2)    # '第三节课_泡吧_第一节课': 0.2
generate_state_action_state_prob_dict(P, S[3], A[3], S[2], prob=0.4)    # '第三节课_泡吧_第二节课': 0.4
generate_state_action_state_prob_dict(P, S[3], A[3], S[3], prob=0.4)    # '第三节课_泡吧_第三节课': 0.4
# 生成基于状态和行为的奖励函数Rsa
generate_state_action_reward_dict(R, S[0], A[0], -1)    # '浏览手机中_浏览手机': -1
generate_state_action_reward_dict(R, S[0], A[2],  0)    # '浏览手机中_离开浏览': 0
generate_state_action_reward_dict(R, S[1], A[0], -1)    # '第一节课_浏览手机': -1
generate_state_action_reward_dict(R, S[1], A[1], -2)    # '第一节课_学习': -2
generate_state_action_reward_dict(R, S[2], A[1], -2)    # '第二节课_学习': -2
generate_state_action_reward_dict(R, S[2], A[4],  0)    # '第二节课_退出学习': 0
generate_state_action_reward_dict(R, S[3], A[1], 10)    # '第三节课_学习': 10
generate_state_action_reward_dict(R, S[3], A[3], +1)    # '第三节课_泡吧': 1
# 生成策略分布
# 状态的价值是基于给定策略，需要事先指定策略pi，这里使用均一随机策略，即某状态下所有可能的行为被选择的概率相等，每一状态可选行为的概率为0.5，初始条件下设置所有状态的价值均为0
# 策略pi：某一状态下的行为集合的概率分布pi(a|s) = P(A_t=a|S_t=s)
generate_pi_dict(Pi, S[0], A[0], 0.5)   # '浏览手机中_选择_浏览手机_概率': 0.5
generate_pi_dict(Pi, S[0], A[2], 0.5)   # '浏览手机中_选择_离开浏览_概率': 0.5
generate_pi_dict(Pi, S[1], A[0], 0.5)   # '第一节课_选择_浏览手机_概率': 0.5
generate_pi_dict(Pi, S[1], A[1], 0.5)   # '第一节课_选择_学习_概率': 0.5
generate_pi_dict(Pi, S[2], A[1], 0.5)   # '第二节课_选择_学习_概率': 0.5
generate_pi_dict(Pi, S[2], A[4], 0.5)   # '第二节课_选择_退出学习_概率': 0.5
generate_pi_dict(Pi, S[3], A[1], 0.5)   # '第三节课_选择_学习_概率': 0.5
generate_pi_dict(Pi, S[3], A[3], 0.5)   # '第三节课_选择_泡吧_概率': 0.5

# 基于给定的MDP和状态价值V的条件下计算某一状态s下采取行为a的价值q(s, a)
# 设定初始状态价值为0
V = {}
def compute_q(MDP, V, s, a):
    assert isinstance(V, dict)
    '''给定MDP和价值函数V， 计算给定s, a下的状态行为对价值函数'''
    S, A, P, R, gamma = MDP
    qsa = 0.
    sa = '_'.join([s, a])   # 拼接字段
    r_sa = R.get(sa, 0)    # 获取的即时奖励
    for s_prime in S:
        sas = '_'.join([s, a, s_prime])   # 拼接字段
        # 获取转移概率
        qsa += P.get(sas, 0) * V.get(s_prime, 0)
    qsa = r_sa + qsa    # (即时奖励 + 转移概率 * 下一个状态的状态价值)
    return qsa

# 给定策略，计算某一状态的价值
def compute_v(MDP, V, Pi, s):
    '''给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值'''
    S, A, P, R, gamma = MDP
    vs = 0.
    for a in A:
        # sa = '_'.join([s, a])
        sa = '_'.join([s, '选择', a, '概率'])
        vs += Pi.get(sa, 0) * compute_q(MDP, V, s, a)
    return vs

# 根据当前策略使用回溯法来更新状态价值
def update_V(MDP, V, Pi):
    '''给定MDP和一个策略，更新该策略下的价值函数V'''
    S, A, P, R, gamma = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[s] = compute_v(MDP, V_prime, Pi, s)

    return V_prime

# 策略评估，得到该策略下最终的状态价值
def policy_evaluate(MDP, V, Pi, n):
    '''使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值，初始时价值为V
    '''
    for i in range(n):
        V = update_V(MDP, V, Pi)
    return V

MDP = (S, A, P, R, gamma)
V = policy_evaluate(MDP, V, Pi, 100)
# 验证状态在某策略下的价值
v = compute_v(MDP, V, Pi, "第三节课")
print("第三节课在当前策略下的价值为:{:.2f}".format(v))

# 不同策略得到的最终状态价值不一样， 最优策略的最优状态价值。此时状态价值将是该状态在所有行为价值中的最大值
def compute_v_from_maxq(MDP, V, s):
    '''根据一个状态的下所有可能的行为价值中最大一个来确定当前状态价值'''
    S, A, P, R, gamma = MDP
    max_q = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa > max_q:
            max_q = qsa

    return max_q


# 得到最优策略和最优状态价值
def update_V_without_pi(MDP, V):
    '''在不依赖策略的情况下直接通过后续状态的价值来更新状态价值'''
    S, A, P, R, gamma = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[s] = compute_v_from_maxq(MDP, V_prime, s)
    return V_prime

# 价值迭代
def value_iterate(MDP, V, n):
    '''价值迭代
    '''
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


V = {}
# 通过价值迭代得到最优状态价值及
V_star = value_iterate(MDP, V, 4)

# 验证最优行为价值
s, a = "第三节课", "泡吧"
q = compute_q(MDP, V_star, "第三节课", "泡吧")
print("在状态{}选择行为{}的最优价值为:{:.2f}".format(s,a,q))
print(MDP)
print(Pi)
