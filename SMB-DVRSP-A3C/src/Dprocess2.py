"""
@author: Viet Nguyen <nhviet1009@gmail.com>
record rewards by time steps
"""
import datetime
import math
import os
import timeit
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import xlwt
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from src.env import create_train_env
from src.model import ActorCritic


def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    global_step = 0
    curr_episode = 0

    """记录数据"""
    curr_time = datetime.datetime.now()
    timestamp = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    filename = opt.data_label + '_' + timestamp + '.xlsx'
    filepath = 'exp_data_by_step/World' + str(opt.world) + '_Stage' + str(opt.stage) + '/'
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    if save:
        """进程0初始化"""
        wb = xlwt.Workbook()
        sh = wb.add_sheet('sheet1')
        sh.write(0, 0, 'TIME STEP')
        sh.write(0, 1, 'ENV REWARD')
        sh.write(0, 2, 'SF REWARD')
        sh.write(0, 3, 'VR REWARD')
        sh.write(0, 4, 'FINAL REWARD')
        sh.write(0, 5, 'DISTANCE')
        sh.write(0, 6, 'FLAG GET')
        sh.write(0, 7, 'GAME NUM')
    """全部进程初始化"""
    flag_get_list = []
    x_pos_list = []
    game_count = 0
    flag_first_get = 0
    DED = []
    TED = []

    while True:
        if save:
            # todo: 读取进程0的info信息
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            # print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            gpu_or_cpu = "GPU"
        else:
            gpu_or_cpu = "CPU"

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, info = env.step(action)

            """获取最大时间长度"""
            if save and curr_step == 1:
                time_max = info['time']

            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True
            if done:
                """存储DED和TED"""
                DED.append(int(info['x_pos']))
                TED.append(curr_step)
                game_count += 1
                flag_get_list.append(int(info['flag_get']))
                x_pos_list.append(info["x_pos"])
                flag = 0
                if game_count >= 100:
                    for i in flag_get_list[-100:]:
                        if i == 1:
                            flag += 1
                    flag_get_rate = flag / 100
                    avg_x_pos = sum(x_pos_list[-100:]) / 100

                """记录数据"""
                if save:
                    """记录数据"""
                    curr_time = datetime.datetime.now()
                    timestamp = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
                    filename = opt.data_label + '_' + timestamp + '.xlsx'
                    filepath = 'exp_data_by_step/World' + str(opt.world) + '_Stage' + str(opt.stage) + '/'
                    if not os.path.isdir(filepath):
                        os.makedirs(filepath)
                    """进程0初始化"""
                    wb = xlwt.Workbook()
                    sh = wb.add_sheet('sheet1')
                    sh.write(0, 0, 'TIME STEP')
                    sh.write(0, 1, 'ENV REWARD')
                    sh.write(0, 2, 'SF REWARD')
                    sh.write(0, 3, 'VR REWARD')
                    sh.write(0, 4, 'FINAL REWARD')
                    sh.write(0, 5, 'DISTANCE')
                    sh.write(0, 6, 'FLAG GET')
                    sh.write(0, 7, 'GAME NUM')

                    total_step = (curr_episode - 1) * opt.num_local_steps + curr_step

                    per = total_step / opt.num_global_steps * 100
                    time_use = timeit.default_timer() - start_time
                    time_left = time_use * (100 - per) / per / 60
                    step_left = opt.num_global_steps - total_step
                    if game_count == 1:
                        time_last10 = start_time
                        step_use10 = 0
                        step_last10 = 0
                        time_left10_h = 99
                        time_left10_m = 99
                        time_left10_s = 99
                    if game_count % 10 == 0:
                        time_use10 = timeit.default_timer() - time_last10
                        step_use10 = total_step - step_last10
                        time_last10 = timeit.default_timer()
                        step_last10 = total_step
                    if game_count == 10:
                        time_left10 = time_use10 / step_use10 * step_left
                    if game_count > 10:
                        time_left10 = (time_use10 / step_use10 * step_left) * 0.1 + time_left10 * 0.9
                        time_left10_h = int(time_left10 // 3600)
                        time_left10_m = int(time_left10 % 3600 // 60)
                        time_left10_s = int(time_left10 % 3600 % 60)

                    print("\r【BY TIME STEPS】W{}S{}-{}使用{}运行生成了【{}】条数据，用时:{:.2f}秒，进度: {:.2f}%，预计{}小时{}分{}秒后完成".format(
                        opt.world, opt.stage, filename, gpu_or_cpu, game_count, time_use, per, time_left10_h,
                        time_left10_m, time_left10_s), end='')
                    if int(info['flag_get']) == 1 and flag_first_get == 0:
                        first_get_step = total_step
                        flag_first_get = 1
                    if game_count > 100 and flag_first_get == 1:
                        print("，最近100局平均探索距离为{}，通关率为{:.2f},首次通关step为{}".format(avg_x_pos, flag_get_rate,
                                                                               first_get_step), end='')
                    elif game_count <= 100 and flag_first_get == 1:
                        print("，首次通关step为{}".format(first_get_step), end='')
                    else:
                        print("，尚未通关T^T", end='')

                global_step += curr_step
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    state = state.cuda()

            """修改step奖励,采样区大小"""
            vs = 10  # Velocity Reward size
            ss = 10  # Stagnation Fine size
            sf_reward = 1
            vr_reward = 1
            delta_reward = 0

            """静止惩罚SF"""
            if opt.use_SF and game_count > ss:
                D_DED = np.var(DED[-ss:])
                D_TED = np.var(TED[-ss:])
                if D_DED + D_TED == 0 and flag_get_rate < ss / 100:
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

            if game_count > 100 and flag_get_rate > 0.9:
                delta_reward = 0

            """速度奖励VR"""
            if opt.use_VR and game_count > vs:
                Base_Velocity = sum(DED[-vs:]) / sum(TED[-vs:])
                vr_reward = int(info['x_pos']) / (curr_step + 1) / Base_Velocity - 0.05  # 0.05
                if vr_reward > 1.5:  # 2
                    vr_reward = 1.5
                elif vr_reward < 0.8:  # 0.5
                    vr_reward = 0.8
            reward_env = reward
            reward = (reward + delta_reward) * vr_reward
            if save:
                step = global_step + curr_step
                sh.write(step, 0, step)
                sh.write(step, 1, reward_env)
                sh.write(step, 2, delta_reward)
                sh.write(step, 3, vr_reward)
                sh.write(step, 4, reward)
                sh.write(step, 5, int(info['x_pos']))
                sh.write(step, 6, int(info['flag_get']))
                sh.write(step, 7, game_count)
                wb.save(filepath + filename)

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
