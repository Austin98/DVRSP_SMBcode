import math
import os
import pickle

import numpy as np
import xlrd
import xlwt


def smooth_curve(data: list, smooth_level: float):
    """
    平滑曲线函数
    :param data: 原始数据
    :param smooth_level: 越大越平滑 (0,1) 原始数据不做处理设置为0 均值设置为1
    :return: list
    """
    if smooth_level == 1:
        iv_data = []
        for i in range(len(data)):
            iv_data.append(np.mean(data))
        return iv_data
    smooth_data = [data[0]]
    for i in range(len(data)):
        next_num = data[i] * (1 - smooth_level) + smooth_data[i] * smooth_level
        smooth_data.append(next_num)
    return smooth_data[1:]


def read_data_from_xls(xls_path: str, row_num: int, data_num=10000):
    """
    从xls表读数据 默认第一行不读
    :param xls_path: xls文件路径
    :param row_num: 读取的列数
    :param data_num: 数据个数
    :return: list
    """
    # if xls_path[-5:] == ".xlsx":
    #     xls_path = xls_path[:-1]
    # elif xls_path[-4:] != ".xls":
    #     xls_path += ".xls"
    sheet_num = 1
    first_row_num = 1
    x = xlrd.open_workbook(xls_path)
    sh = x.sheet_by_index(sheet_num - 1)
    iv_data = sh.col_values(row_num, first_row_num, data_num + 1)
    return iv_data


def gen_file_path(file_path):
    """
    生成文件路径,若路径不存在则生成路径
    :param file_path: 文件路径，可以用列表一次操作多个路径
    :return:
    """
    if type(file_path) == str:
        iv_file_path = file_path
        if not os.path.isdir(iv_file_path):
            os.makedirs(iv_file_path)
    elif type(file_path) == list:
        for iv_file_path in file_path:
            if not os.path.isdir(iv_file_path):
                os.makedirs(iv_file_path)


def gen_xls_from_pkl(pkl_path: str, xls_path: str, log_num=10, divide_num=1):
    """
    从pkl文件生成xls文件
    :param divide_num: 数据处理的被除数 默认为1
    :param pkl_path: 三个pkl文件的上级目录，一般为memory文件夹
    :param xls_path: 生成的xls路径
    :param log_num: 数据处理的底数 默认为10
    :return: 绘图调用rol_num：4-坦克 5-战车 6-人员 7-加和
    """
    iv_x = xlwt.Workbook('encoding = utf-8')
    iv_sh = iv_x.add_sheet("sheet1", cell_overwrite_ok=True)
    for iv_ic in range(8):
        iv_sh.col(iv_ic).width = 12 * 256
    if pkl_path[-1] == '/':
        pkl_path = pkl_path[:-1]
    if xls_path[-5:] == '.xlsx':
        xls_path = xls_path[:-1]
    elif xls_path[-4:] != '.xls':
        xls_path += '.xls'
    pkl_files = [pkl_path + "/Tankarrivaltimes.pkl",
                 pkl_path + "/Cararrivaltimes.pkl",
                 pkl_path + "/Peoplearrivaltimes.pkl"]
    iv_sh.write(0, 0, "地图格子编号")
    iv_sh.write(0, 1, "坦克到达次数")
    iv_sh.write(0, 2, "战车到达次数")
    iv_sh.write(0, 3, "人员到达次数")
    iv_sh.write(0, 4, "坦克数据处理")
    iv_sh.write(0, 5, "战车数据处理")
    iv_sh.write(0, 6, "人员数据处理")
    iv_sh.write(0, 7, "数据处理加和")

    iv_j_total = []
    for i in range(299):
        iv_j_total.append(0)
    for iv_i in range(3):
        iv_pkl = pkl_files[iv_i]
        iv_f = open(iv_pkl, 'rb')
        iv_context = pickle.load(iv_f)
        iv_f.close()
        iv_a = iv_context.__str__()
        iv_b = iv_a[:-1]
        iv_c = iv_b.split(",")
        for iv_ib in range(len(iv_c)):
            iv_d = iv_c[iv_ib].split(":")
            iv_e = int(iv_d[0][2:-1])
            iv_f = int(iv_d[1][1:]) / divide_num
            iv_sh.write(iv_ib + 1, 0, iv_e)
            iv_sh.write(iv_ib + 1, iv_i + 1, iv_f)
            iv_j = math.log(iv_f + 1, log_num)
            iv_j_total[iv_ib] += iv_j
            iv_sh.write(iv_ib + 1, iv_i + 4, iv_j)
            iv_sh.write(iv_ib + 1, 7, iv_j_total[iv_ib])
    iv_x.save(xls_path)


def sort_data(data: list, sort_type=1):
    """
    :param data: 原始数据
    :param sort_type: 0：升序 1：降序
    :return: list
    """
    iv_data = np.sort(data).tolist()
    if sort_type:
        iv_data = iv_data[::-1]
    return iv_data


def gen_gray_level(current_data=0, dark=0.0, light=1.0, max_num=0):
    """
    图像灰阶生成
    :param current_data: 列表中读取的数据
    :param dark: 最深灰度 0-黑 1-白
    :param light: 最亮灰度
    :param max_num: 列表中数据的最大值（外部传入）【必需】
    :return: 处理后的RGB灰阶数值
    """
    if 0 < current_data <= max_num:
        color = light - current_data * (light - dark) / max_num
        return color, color, color
    elif current_data == 0:
        return 1, 1, 1


def cal_win_rate(data: list, cal_num, cut_score=0, bool_include_draw=True):
    """
    得分转化为胜率 目前只能分片显示
    todo：平滑胜率曲线
    :param data:  分数
    :param cal_num:  每组分片数据量
    :param cut_score:  截断分数，以判断胜败 默认为0
    :param bool_include_draw: 胜利是否包含平局 默认包含
    :return:
    """
    iv_data = data
    group_num = int(len(data) / cal_num)
    for i in range(group_num):
        win_num = 0
        win_rate = 0
        for ib in range(cal_num):
            if bool_include_draw:
                if data[i * cal_num + ib] >= cut_score:
                    win_num += 1
                    win_rate = win_num / cal_num
            else:
                if data[i * cal_num + ib] > cut_score:
                    win_num += 1
                    win_rate = win_num / cal_num
        for ic in range(cal_num):
            iv_data[i * cal_num + ic] = win_rate
    remain_num = len(data) % cal_num
    if remain_num != 0:
        for ie in range(remain_num):
            iv_data[-ie - 1] = iv_data[-remain_num - 1]
    return iv_data


def data_all_same(num: float, length: int):
    """
    生成一组相同数据的列表
    :param num:  填充的数据
    :param length:  列表长度
    :return:
    """
    iv_data = []
    for i in range(length):
        iv_data.append(num)
    return iv_data


def zoom_data(data: list, zoom_ratio: float):
    """
    缩放列表
    :param data:原始列表数据
    :param zoom_ratio: 缩放比例
    :return:
    """
    new_data = []
    new_data_len = int(len(data) * zoom_ratio)
    for i in range(new_data_len):
        data_index = int(i / zoom_ratio)
        new_data.append(data[data_index])
    return new_data


def cal_error(data: list, cal_num=16, error_size=1):
    """
    生成误差带上下界数据
    :param data: 原始列表数据
    :param cal_num: 误差计算窗口
    :param error_size: 误差放大倍数，正值为上界，负值为下界
    :return:
    """
    iv_data_list = []
    for i in range(len(data) - cal_num):
        iv_sample = data[i:i + cal_num]
        iv_data = data[i] + error_size * np.std(iv_sample, ddof=1) / math.sqrt(cal_num)
        iv_data_list.append(iv_data)
    for i in range(cal_num):
        iv_sample = data[-cal_num:]
        iv_data = data[len(data) - cal_num + i] + error_size * np.std(iv_sample, ddof=1) / math.sqrt(cal_num)
        iv_data_list.append(iv_data)
    return iv_data_list
