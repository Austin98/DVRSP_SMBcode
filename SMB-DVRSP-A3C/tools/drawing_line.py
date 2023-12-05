import matplotlib.pyplot as plt
from utils import *

dn = 10000
sc = 0.995
rn = 1
x_label = 'Game Numbers'
if rn == 1:
    y_label = 'x_pos'
elif rn == 2:
    y_label = "time"
elif rn == 3:
    y_label = "score"
data1 = read_data_from_xls('/home/lab811/桌面/PPO_improved/exp_data/PPO_iv04(BASE)/1_1/2022-05-31 06:46:30.768399.xls', row_num=rn,
                           data_num=dn)
data1[0] = 0
data1 = smooth_curve(data1, sc)
data1_label = 'BASE'

data2 = read_data_from_xls('/home/lab811/桌面/PPO_improved/exp_data/PPO_iv04(only SF10)/1_1/2022-05-30 21:51:25.344744.xls', row_num=rn,
                           data_num=dn)
data2[0] = 0
data2 = smooth_curve(data2, sc)
data2_label = 'SF10'

data3 = read_data_from_xls('/home/lab811/桌面/PPO_improved/exp_data/PPO_iv04(ONLY SF50)/1_1/2022-05-31 06:47:22.667376.xls', row_num=rn,
                           data_num=dn)
data3[0] = 0
data3 = smooth_curve(data3, sc)
data3_label = 'SF50'

data4 = read_data_from_xls('/home/lab811/桌面/PPO_improved/exp_data/PPO_iv04(only SF100)/1_1/2022-05-30 21:51:44.657153.xls', row_num=rn,
                           data_num=dn)
data4[0] = 0
data4 = smooth_curve(data4, sc)
data4_label = 'SF100'

data5 = read_data_from_xls('/home/lab811/桌面/PPO_improved/exp_data/PPO_iv04(ONLY SF50)/1_1/2022-05-31 21:37:05.427128.xls', row_num=rn,
                           data_num=dn)
data5[0] = 0
data5 = smooth_curve(data5, sc)
data5_label = 'SF50-2'

data6 = read_data_from_xls('/home/lab811/桌面/PPO_improved/exp_data/PPO_iv04(BASE)/1_1/2022-05-31 21:24:06.770696.xls', row_num=rn,
                           data_num=dn)
data6[0] = 0
data6 = smooth_curve(data6, sc)
data6_label = 'BASE-2'
#
# data7 = read_data_from_xls('PPO_base_2_3/2022-05-10 21:39:32.260329.xls', row_num=rn,
#                            data_num=dn)
# data7[0] = 0
# data7 = smooth_curve(data7, sc)
# data7_label = 'DATA2-3'
#
# data8 = read_data_from_xls('PPO_base_2_4/2022-05-10 21:39:43.509352.xls', row_num=rn,
#                            data_num=dn)
# data8[0] = 0
# data8 = smooth_curve(data8, sc)
# data8_label = 'DATA2-4'

plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 英文字体Arial, 中文字体：SimHei

x1 = np.array(range(1, len(data1) + 1))
y1 = np.array(data1)
x2 = np.array(range(1, len(data2) + 1))
y2 = np.array(data2)
x3 = np.array(range(1, len(data3) + 1))
y3 = np.array(data3)
x4 = np.array(range(1, len(data4) + 1))
y4 = np.array(data4)
x5 = np.array(range(1, len(data5) + 1))
y5 = np.array(data5)
x6 = np.array(range(1, len(data6) + 1))
y6 = np.array(data6)
# x7 = np.array(range(1, len(data7) + 1))
# y7 = np.array(data7)
# x8 = np.array(range(1, len(data8) + 1))
# y8 = np.array(data8)

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、...
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(10, 5))
plt.plot(x1, y1, marker=',', linestyle='-', color="b", label=data1_label, linewidth=2)
plt.plot(x2, y2, marker=',', linestyle='-', color="g", label=data2_label, linewidth=2)
plt.plot(x3, y3, marker=',', linestyle='-', color="r", label=data3_label, linewidth=2)
plt.plot(x4, y4, marker=',', linestyle='-', color="c", label=data4_label, linewidth=2)
plt.plot(x5, y5, marker=',', linestyle='-', color="m", label=data5_label, linewidth=2)
plt.plot(x6, y6, marker=',', linestyle='-', color="y", label=data6_label, linewidth=2)
# plt.plot(x7, y7, marker=',', linestyle='-', color="k", label=data7_label, linewidth=2)
# plt.plot(x8, y8, marker=',', linestyle='-', color="w", label=data8_label, linewidth=2)

# note：设置边框和背景网格线
# ax = plt.gca()  # 去边框
# ax.spines['top'].set_visible(False)  # 去掉上边框
# ax.spines['right'].set_visible(False)  # 去掉右边框
plt.grid(linestyle="-")  # 设置背景网格线
# note:手动设置x轴坐标值
# x_actual = [0, 20, 40, 60, 80, 100]
# x_mapping = [0, 1000, 2000, 3000, 4000, 5000]
# plt.xticks(x_actual, x_mapping, fontsize=10, fontweight='bold')
# note:手动设置y轴坐标值
# y_actual = [0, 20, 40, 60, 80, 100]
# y_mapping = [0, 0.2, 0.4, 0.6, 0.8, 1]
# plt.xticks(y_actual, y_mapping, fontsize=12, fontweight='bold')
# note:设置图标和轴标
# fig_title = "对比图"
# plt.title(fig_title, fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel(x_label, fontsize=10, fontweight='bold')
plt.ylabel(y_label, fontsize=10, fontweight='bold')
# note:设置坐标值范围
# plt.xlim(0, 5000)
# plt.ylim(0, 1)
# note:设置图例
plt.legend(loc="best", numpoints=1)
legend_text = plt.gca().get_legend().get_texts()
plt.setp(legend_text, fontsize=10)  # 设置图例字体的大小
plt.show()
