#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 0021 17:01
# @Author  : skydm
# @email   : 
# @File    : dynamic_program.py
# @Software: PyCharm
# @desc    :

'''
动态规划的使用场景：
1） 复杂问题的最优解由若干个小问题的最优解构成，寻找子问题的最优解得到复杂问题的最优解
2） 子问题在复杂问题内重复出现，使子问题的解可以被存储重复利用

马尔科夫决策过程：
贝尔曼方程把问题递归为求解子问题
价值函数相当于存储了一些子问题的解

本节求解该方格世界在给定策略下的(状态)价值函数，也就是求解在给定策略下，该方格世界里每一个状态的价值。
'''
import math
S = [i for i in range(16)]  # 状态(4*4方格世界)
A = ['n', 'e', 's', 'w']    # 行为空间

ds_actions = {'n': -4, 'e': 1, 's': 4, 'w': -1}     # 行为对状态的改变(环境动力学特征)

def dynamics(s, a):
    '''
    :param s:   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    :param a:   ['n','e','s','w'] 表示北，东， 南， 西 (按照上北下南， 左西右东的概念)
    :return:
    tuple (s_prime, reward, is_end)
    s_prime： 后续状态
    reward：奖励值
    is_end: 是否进入终止状态
    '''
    s_prime = s
    if (s % 4 == 0 and a == 'w') or (s < 4 and a == 'n') or ((s + 1) % 4 == 0 and a == 'e') or (s > 11 and a == 's') or s in [0, 15]:
        # 如果处于边界情况，遇到上述
        pass
    else:
        ds = ds_actions[a]  # 行为对状态的改变(动力学特征)
        s_prime = s + ds    # 新的状态

    reward = 0 if s in [0, 15] else -1  # 落入状态获取的奖励
    is_end = True if s in [0, 15] else False
    return s_prime, reward, is_end


def P(s, a, s1):  # 状态转移概率
    # 任何试图离开方格世界的动作其位置将不会发生改变，其余条件下将100%地转移到动作指向的位置；
    s_prime, _, _ = dynamics(s, a)  # 进入下一个状态
    return s1 == s_prime


def R(s, a):  # 奖励函数
    # 任何在非终止状态间的转移得到的即时奖励均为-1，进入终止状态即时奖励为0
    _, r, _ = dynamics(s, a)    # 进入到下一个状态获取的奖励
    return r

gamma = 1.00
MDP = S, A, R, P, gamma

# 开始建立策略(均一随机策略和贪婪策略)
def uniform_random_pi(MDP=None, V=None, s=None, a=None):
    '''均一随机策略'''
    S, A, R, P, gamma = MDP
    n = len(A)
    return 0 if n == 0 else 1.0/n

def greedy_pi(MDP, V, s, a):
    '''贪婪策略, 对于状态s，采取行动a的概率'''
    S, A, R, P, gamma = MDP
    max_v, a_max_v = -float('inf'), []  # 统计后续状态的最大价值以及到达该状态的行为(注意可能不只一个)
    for a_opt in A:
        # 获取后续状态的价值
        s_prime, reward, is_end = dynamics(s, a_opt)
        # 获取s_prime的价值
        v_s_prime = V.get(s_prime)
        if v_s_prime > max_v:   # 如果最大，则记录
            max_v = v_s_prime
            a_max_v = [a_opt]
        elif (v_s_prime == max_v):  # 如果相等，则添加
            a_max_v.append(a_opt)

    # 生成策略分布
    n = len(a_max_v)
    if n == 0:
        return 0.0
    return 1.0 / n if a in a_max_v else 0.0


def get_pi(Pi, s, a, MDP=None, V=None):
    return Pi(MDP, V, s, a)


# 迭代法策略评估： 计算给定策略下状态价值函数
# 策略迭代： 给定一个策略pi时，得到基于该策略的价值函数，然后基于产生的价值函数得到贪婪策略，基于新的策略得到新的价值函数，
# 并产生新的贪婪策略，如此重复循环迭代，得到最终最优价值函数和最优策略。是个收敛的过程。
# 价值迭代：减少迭代次数， 因为最优策略可能是一样的
def compute_q(MDP, V, s, a):
    '''给定MDP， 价值函数V，给定状态行为对(s, a)。计算q_sa'''
    S, A, R, P, gamma = MDP
    qsa = 0.
    # 获得即时奖励
    s_prime, reward, is_end = dynamics(s, a)
    for s_prime in S:
        # 状态s采取行动a到达状态s_prime的概率
        qsa += gamma * P(s, a, s_prime) * V.get(s_prime, 0)
    qsa = reward + qsa
    return qsa

def compute_v(MDP, V, Pi, s):
    '''给定MDP， 依据某一策略Pi和当前的状态价值函数V来计算某状态s的价值'''
    S, A, R, P, gamma = MDP
    vs = 0.
    # 状态s下采取策略pi，即行动a的概率
    for a in A:
        # 对应采取行动a的概率
        vs += get_pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)
    return vs

def update_V(MDP, V, Pi):
    '''给定MDP和策略Pi， 更新该策略的价值函数V'''
    S, A, R, P, gamma = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[s] = compute_v(MDP, V, Pi, s)
    return V_prime

def policy_evaluate(MDP, V, Pi, n):
    '''进行策略评估， MDP在给定策略Pi下的状态价值函数'''
    for i in range(n):
        V = update_V(MDP, V, Pi)
    return V

def policy_iterate(MDP, V, Pi, n, m):
    '''使用贪婪策略'''
    for i in range(m):
        V = policy_evaluate(MDP, V, Pi, n)
        Pi = greedy_pi  # 第一次迭代产生的价值函数后使用贪婪策略
    return V

def compute_v_from_maxq(MDP, V, s):
    '''根据一个状态的下所有可能的行为价值中最大一个来确定当前状态价值'''
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa > v_s:
            v_s = qsa

    return v_s

def update_V_without_pi(MDP, V):
    '''在不依赖策略的情况下直接通过后续状态的价值来更新状态价值'''
    S, A, R, P, gamma = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[s] = compute_v_from_maxq(MDP, V_prime, s)
    return V_prime

def value_iterate(MDP, V, n):
    '''价值迭代'''
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


V = [0 for _ in range(16)]
V = dict(zip(S, V))
print(V)
# 策略评估，在确定策略下，最终的状态价值函数
V_pi = policy_evaluate(MDP, V, uniform_random_pi, 100)
i = 0
for s, value in V_pi.items():
    i = i + 1
    if i % 4 != 0:
        print(s, ':', math.floor(value), end='\t')
    else:
        print(s, ':', math.floor(value))

# 策略迭代(贪婪策略)
V_pi = policy_iterate(MDP, V, greedy_pi, 1, 100)
print(V_pi)

# 价值迭代
V_star = value_iterate(MDP, V, 4)
print(V_star)

# 观察最优状态下对应的最优策略
def greedy_policy(MDP, V, s):
    S, A, P, R, gamma = MDP
    max_v, a_max_v = -float('inf'), []
    for a_opt in A: # 统计后续状态的最大价值以及到达到达该状态的行为（可能不止一个）
        s_prime, reward, _ = dynamics(s, a_opt)
        v_s_prime = V[s_prime]  # 获取下一个状态的价值
        if v_s_prime > max_v:
            max_v = v_s_prime
            a_max_v = a_opt
        elif(v_s_prime == max_v):
            a_max_v += a_opt
    return str(a_max_v)


def display_policy(policy, MDP, V):
    S, A, P, R, gamma = MDP
    for i in range(16):
        print('{0:^6}'.format(policy(MDP, V, S[i])), end = " ")
        if (i+1) % 4 == 0:
            print("")
    print()

display_policy(greedy_policy, MDP, V_star)