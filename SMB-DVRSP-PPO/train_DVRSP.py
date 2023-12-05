"""
@Original author: Viet Nguyen <nhviet1009@gmail.com>
DVRSP author: Qiyang Zhang <qiyangz@foxmail.com>
"""

import datetime
import math
import timeit

import xlwt

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--use_VR", type=bool, default=True)
    parser.add_argument("--use_SF", type=bool, default=True)
    parser.add_argument("--data_label", type=str, default="PPO_DVRSF")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=72e4)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
        gpu_or_cpu = "GPU"
    else:
        gpu_or_cpu = "CPU"
    curr_episode = 0


    """记录数据"""
    start_time = timeit.default_timer()
    curr_time = datetime.datetime.now()
    timestamp = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    filename = opt.data_label + '_' + timestamp + '.xls'
    filepath = 'exp_data/World' + str(opt.world) + '_Stage' + str(opt.stage) + '/'
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    wb = xlwt.Workbook()
    sh = wb.add_sheet('sheet1')
    sh.write(0, 0, 'game_num')
    sh.write(0, 1, 'curr_step')
    sh.write(0, 2, 'total_step')
    sh.write(0, 3, 'coins')
    sh.write(0, 4, 'flag_get')
    sh.write(0, 5, 'life')
    sh.write(0, 6, 'score')
    sh.write(0, 7, 'stage')
    sh.write(0, 8, 'status')
    sh.write(0, 9, 'time')
    sh.write(0, 10, 'world')
    sh.write(0, 11, 'x_pos')
    sh.write(0, 12, 'y_pos')
    """初始化"""
    flag_get_list = []
    x_pos_list = []
    game_count = [0, 0, 0, 0, 0, 0]
    curr_step = [0, 0, 0, 0, 0, 0]
    flag_first_get = 0
    last_game_num = -1
    DED = [[], [], [], [], [], []]
    TED = [[], [], [], [], [], []]


    while True:
        # if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []

        """全部进程初始化"""
        if curr_episode == 1:
            """code"""


        for _ in range(opt.num_local_steps):
            for i in range(len(curr_step)):
                curr_step[i] += 1
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            # todo:读取info
            # todo:改reward

            # if curr_step > opt.num_global_steps:
            #     done = True


            """获取最大时间长度"""
            if curr_step[0] == 1:
                time_max = int(info[0]['time'])
            """记录数据"""
            if done[0] and last_game_num != game_count[0]:
                last_game_num = game_count[0]
                total_step = (curr_episode - 1) * opt.num_local_steps + curr_step[0]
                sh.write(game_count[0]+1, 0, game_count[0]+1)
                sh.write(game_count[0]+1, 1, curr_step[0])
                sh.write(game_count[0]+1, 2, total_step)
                sh.write(game_count[0]+1, 3, info[0]['coins'])
                sh.write(game_count[0]+1, 4, int(info[0]['flag_get']))
                sh.write(game_count[0]+1, 5, int(info[0]['life']))
                sh.write(game_count[0]+1, 6, info[0]['score'])
                sh.write(game_count[0]+1, 7, int(info[0]['stage']))
                sh.write(game_count[0]+1, 8, info[0]['status'])
                sh.write(game_count[0]+1, 9, time_max - int(info[0]['time']))
                sh.write(game_count[0]+1, 10, int(info[0]['world']))
                sh.write(game_count[0]+1, 11, int(info[0]['x_pos']))
                sh.write(game_count[0]+1, 12, int(info[0]['y_pos']))
                wb.save(filepath + filename)

                per = total_step / opt.num_global_steps * 100
                time_use = timeit.default_timer() - start_time
                time_left = time_use * (100 - per) / per / 60
                step_left = opt.num_global_steps - total_step
                if game_count[0] == 0:
                    time_last10 = start_time
                    step_use10 = 0
                    step_last10 = 0
                    time_left10_h = 99
                    time_left10_m = 99
                    time_left10_s = 99
                if game_count[0] % 10 == 0 and game_count[0] != 0:
                    time_use10 = timeit.default_timer() - time_last10
                    step_use10 = total_step - step_last10
                    time_last10 = timeit.default_timer()
                    step_last10 = total_step
                if game_count[0] == 10:
                    time_left10 = time_use10 / step_use10 * step_left
                if game_count[0] > 10:
                    time_left10 = (time_use10 / step_use10 * step_left) * 0.1 + time_left10 * 0.9
                    time_left10_h = int(time_left10 // 3600)
                    time_left10_m = int(time_left10 % 3600 // 60)
                    time_left10_s = int(time_left10 % 3600 % 60)

                print("\rW{}S{}-{}使用{}运行生成了【{}】条数据，用时:{:.2f}秒，进度: {:.2f}%，预计{}小时{}分{}秒后完成".format(
                    opt.world, opt.stage, filename, gpu_or_cpu, game_count[0]+1, time_use, per, time_left10_h,
                    time_left10_m, time_left10_s), end='')
                if int(info[0]['flag_get']) == 1 and flag_first_get == 0:
                    first_get_step = total_step
                    flag_first_get = 1
                if game_count[0]+1 > 100 and flag_first_get == 1:
                    print("，最近100局平均探索距离为{}，通关率为{:.2f},首次通关step为{}".format(avg_x_pos, flag_get_rate,
                                                                           first_get_step), end='')
                elif game_count[0]+1 <= 100 and flag_first_get == 1:
                    print("，首次通关step为{}".format(first_get_step), end='')
                else:
                    print("，尚未通关T^T", end='')

            """修改step奖励,采样区大小"""
            vs = 10  # Velocity Reward size
            ss = 10  # Stagnation Fine size
            vr_reward = 1
            delta_reward = 0

            """记录DED & TED"""
            for i in range(len(done)):
                if done[i]:
                    """记录DED & TED"""
                    DED[i].append(int(info[i]['x_pos']))
                    TED[i].append(curr_step[i])
                    game_count[i] += 1
                    flag_get_list.append(int(info[i]['flag_get']))
                    x_pos_list.append(info[i]["x_pos"])
                    flag = 0
                    if game_count[i] >= 100:
                        for i in flag_get_list[-100:]:
                            if i == 1:
                                flag += 1
                        flag_get_rate = flag / 100
                        avg_x_pos = sum(x_pos_list[-100:]) / 100
                    curr_step[i] = 0

            """构造奖励"""
            for i in range(len(reward)):
                vr_reward = 1
                delta_reward = 0
                """静止惩罚SF"""
                if opt.use_SF and game_count[i] > ss:
                    D_DED = np.var(DED[i][-ss:])
                    D_TED = np.var(TED[i][-ss:])
                    if D_DED + D_TED == 0 and flag_get_rate < ss/100:
                        delta_reward = -0.5
                    elif D_DED + D_TED == 0:
                        delta_reward = 0
                    else:
                        sf_reward = 1e6 / (D_TED + D_DED)  # 1e6
                        delta_reward -= (math.log10(sf_reward) - 1) * 0.1  # 0.1
                        if delta_reward > 0.5:
                            delta_reward = 0.5
                        elif delta_reward < -0.5:
                            delta_reward = -0.5

                if game_count[i] > 100 and flag_get_rate > 0.5:
                    delta_reward = 0

                """速度奖励VR"""
                if opt.use_VR and game_count[i] > vs:
                    Base_Velocity = sum(DED[i][-vs:]) / sum(TED[i][-vs:])
                    vr_reward = int(info[i]['x_pos']) / (curr_step[i] + 1) / Base_Velocity - 0.05  # 0.05
                    if vr_reward > 1.5:  # 2
                        vr_reward = 1.5
                    elif vr_reward < 0.8:  # 0.5
                        vr_reward = 0.8

                reward_list = list(reward)
                reward_list[i] = (reward_list[i] + delta_reward) * vr_reward
                reward = tuple(reward_list)



            state = torch.from_numpy(np.concatenate(state, 0))
            done = np.array(done, dtype=np.float32)
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]  # 倒序排序
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)  # 将8*512个数字打乱随机排序
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                #  分16批随机抽取
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        # print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
